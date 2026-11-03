"""Deterministic evidence and completion gate for controlled task runs."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .snapshot import FileTreeSnapshot, capture_tree, materialize_snapshot


DEFAULT_VERIFICATION_EXCLUDE_DIRS = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)


VerificationOutcome = Literal[
    "passed",
    "code_failed",
    "verifier_failed",
    "infrastructure_failed",
    "budget_stopped",
    "acceptance_corrupted",
    "verifier_modified_workspace",
]


@dataclass(frozen=True)
class AcceptanceContract:
    contract_id: str
    required_commands: tuple[str, ...]
    version: int = 1
    acceptance_dir: str | Path | None = None
    acceptance_tree_sha256: str | None = None
    verification_exclude_dirs: tuple[str, ...] = DEFAULT_VERIFICATION_EXCLUDE_DIRS

    def __post_init__(self) -> None:
        # Normalize lexically: verification must not depend on the original
        # directory still existing or resolving to the same files after start.
        if self.acceptance_dir is not None:
            object.__setattr__(
                self,
                "acceptance_dir",
                os.path.abspath(Path(self.acceptance_dir).expanduser()),
            )
        for name in self.verification_exclude_dirs:
            if (
                not isinstance(name, str)
                or name in ("", ".", "..")
                or any(char in name for char in ("/", "\\", "\0"))
            ):
                raise ValueError(
                    f"verification exclusion must be a directory name: {name!r}"
                )
        object.__setattr__(
            self,
            "verification_exclude_dirs",
            tuple(
                sorted(
                    set(DEFAULT_VERIFICATION_EXCLUDE_DIRS)
                    | set(self.verification_exclude_dirs)
                )
            ),
        )


def validate_acceptance_directory(
    directory: str | Path, *, code_dir: str | Path
) -> None:
    base = Path(directory).expanduser().absolute()
    code = Path(code_dir).expanduser().resolve()
    if base.is_relative_to(code) or code.is_relative_to(base):
        raise ValueError(
            f"acceptance_dir must be outside and not contain code_dir: {directory}"
        )
    if not base.is_dir():
        raise ValueError(
            f"acceptance_dir must be an existing nonempty directory: {directory}"
        )
    if any(path.is_symlink() for path in (base, *base.parents)):
        raise ValueError(f"acceptance_dir must not use symlinks: {directory}")
    has_file = False
    for path in base.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"acceptance_dir must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"acceptance_dir requires regular files: {path}")
        has_file = True
    if not has_file:
        raise ValueError(
            f"acceptance_dir must contain at least one regular file: {directory}"
        )


@dataclass(frozen=True)
class VerificationReceipt:
    task_id: str
    task_revision: int
    tree_sha256: str
    contract_id: str
    contract_version: int
    verifier_version: str
    command: str
    exit_code: int
    outcome: VerificationOutcome
    output_digest: str
    acceptance_tree_sha256: str = ""
    verification_exclude_dirs: tuple[str, ...] | None = None
    workspace_changes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.verification_exclude_dirs is not None:
            object.__setattr__(
                self, "verification_exclude_dirs", tuple(self.verification_exclude_dirs)
            )
        object.__setattr__(self, "workspace_changes", tuple(self.workspace_changes))


def _verification_tree(
    root: str | Path, exclude_dirs: tuple[str, ...] | None
) -> FileTreeSnapshot:
    if exclude_dirs is None:
        return capture_tree(root)
    base = Path(root).expanduser().resolve()
    excluded = []
    for path in base.rglob("*"):
        relative = path.relative_to(base)
        if any(part in exclude_dirs for part in relative.parts[:-1]) or (
            path.is_dir() and path.name in exclude_dirs
        ):
            excluded.append(relative.as_posix())
    # capture_tree's exact-path exclusions and checkpoint semantics stay intact.
    return capture_tree(base, exclude=excluded)


def _acceptance_tempdir(code: Path, store: Path | None) -> tempfile.TemporaryDirectory:
    for candidate in (
        Path(tempfile.gettempdir()).resolve(),
        code.parent,
        Path.home().resolve(),
    ):
        if candidate.is_relative_to(code) or (
            store is not None and candidate.is_relative_to(store)
        ):
            continue
        try:
            return tempfile.TemporaryDirectory(
                prefix="horizon-acceptance-", dir=candidate
            )
        except OSError:
            continue
    raise OSError("no temporary directory available outside code_dir and store_dir")


def run_verifier(
    task_id: str,
    root: str | Path,
    contract: AcceptanceContract,
    *,
    verifier_version: str = "local-v1",
    task_revision: int = 0,
    timeout_seconds: float = 300,
    acceptance_snapshot: FileTreeSnapshot | None = None,
    object_dir: str | Path | None = None,
    store_dir: str | Path | None = None,
) -> VerificationReceipt:
    base = Path(root).expanduser().resolve()
    snapshot = _verification_tree(base, contract.verification_exclude_dirs)
    command = " && ".join(contract.required_commands)
    if not command:
        raise ValueError("acceptance contract must contain a required command")
    if timeout_seconds <= 0:
        raise ValueError("verifier timeout must be greater than zero")
    with ExitStack() as cleanup:
        env = None
        try:
            if contract.acceptance_dir is not None:
                if (
                    acceptance_snapshot is None
                    or object_dir is None
                    or not contract.acceptance_tree_sha256
                    or acceptance_snapshot.tree_sha256
                    != contract.acceptance_tree_sha256
                ):
                    raise ValueError(
                        "frozen acceptance snapshot or tree hash missing or mismatched"
                    )
                store = Path(store_dir or Path(object_dir).parent).resolve()
                target = cleanup.enter_context(_acceptance_tempdir(base, store))
                materialize_snapshot(
                    acceptance_snapshot, object_dir=object_dir, target=target
                )
                env = {**os.environ, "HORIZON_ACCEPTANCE_DIR": target}
        except (OSError, ValueError) as exc:
            exit_code, outcome, output = (
                125,
                "acceptance_corrupted",
                f"{type(exc).__name__}: {exc}",
            )
        else:
            exit_code, outcome, output = _execute_command(
                command, base, timeout_seconds, env
            )
    after = _verification_tree(base, contract.verification_exclude_dirs)
    expected = {entry.path: entry for entry in snapshot.entries}
    actual = {entry.path: entry for entry in after.entries}
    changes: list[str] = []
    for path in sorted(expected.keys() | actual.keys()):
        if path not in expected:
            changes.append(f"added:{path}")
        elif path not in actual:
            changes.append(f"removed:{path}")
        elif expected[path] != actual[path]:
            changes.append(f"changed:{path}")
    if snapshot.tree_sha256 != after.tree_sha256 and outcome in (
        "passed",
        "code_failed",
    ):
        outcome = "verifier_modified_workspace"
    return VerificationReceipt(
        task_id=task_id,
        task_revision=task_revision,
        tree_sha256=snapshot.tree_sha256,
        contract_id=contract.contract_id,
        contract_version=contract.version,
        verifier_version=verifier_version,
        command=command,
        exit_code=exit_code,
        outcome=outcome,
        output_digest=hashlib.sha256(output.encode()).hexdigest(),
        acceptance_tree_sha256=contract.acceptance_tree_sha256 or "",
        verification_exclude_dirs=contract.verification_exclude_dirs,
        workspace_changes=tuple(changes),
    )


def _execute_command(
    command: str,
    base: Path,
    timeout_seconds: float,
    env: dict[str, str] | None,
) -> tuple[int, VerificationOutcome, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=base,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
        exit_code = completed.returncode
        if exit_code == 0:
            outcome: VerificationOutcome = "passed"
        elif exit_code in (126, 127):
            outcome = "infrastructure_failed"
        elif exit_code < 0 or exit_code >= 128:
            outcome = "verifier_failed"
        else:
            outcome = "code_failed"
        output = completed.stdout + "\n" + completed.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        outcome = "budget_stopped"
        output = f"{type(exc).__name__}: {exc}"
    except OSError as exc:
        exit_code = 126
        outcome = "infrastructure_failed"
        output = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        exit_code = 125
        outcome = "verifier_failed"
        output = f"{type(exc).__name__}: {exc}"
    return exit_code, outcome, output


def receipt_is_current(
    receipt: VerificationReceipt,
    *,
    task_id: str,
    root: str | Path,
    contract: AcceptanceContract,
    verifier_version: str,
    task_revision: int = 0,
) -> bool:
    return (
        receipt.task_id == task_id
        and receipt.task_revision == task_revision
        and receipt.contract_id == contract.contract_id
        and receipt.contract_version == contract.version
        and receipt.verifier_version == verifier_version
        and receipt.acceptance_tree_sha256 == (contract.acceptance_tree_sha256 or "")
        and (contract.acceptance_dir is None or bool(contract.acceptance_tree_sha256))
        and (
            receipt.verification_exclude_dirs is None
            or receipt.verification_exclude_dirs == contract.verification_exclude_dirs
        )
        and receipt.tree_sha256
        == _verification_tree(root, receipt.verification_exclude_dirs).tree_sha256
        and receipt.exit_code == 0
        and receipt.outcome == "passed"
    )
