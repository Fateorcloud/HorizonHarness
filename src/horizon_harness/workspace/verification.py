"""Deterministic evidence and completion gate for controlled task runs."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .snapshot import capture_tree


VerificationOutcome = Literal[
    "passed",
    "code_failed",
    "verifier_failed",
    "infrastructure_failed",
    "budget_stopped",
]


@dataclass(frozen=True)
class AcceptanceContract:
    contract_id: str
    required_commands: tuple[str, ...]
    version: int = 1


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


def run_verifier(
    task_id: str,
    root: str | Path,
    contract: AcceptanceContract,
    *,
    verifier_version: str = "local-v1",
    task_revision: int = 0,
    timeout_seconds: float = 300,
) -> VerificationReceipt:
    base = Path(root).expanduser().resolve()
    snapshot = capture_tree(base)
    command = " && ".join(contract.required_commands)
    if not command:
        raise ValueError("acceptance contract must contain a required command")
    if timeout_seconds <= 0:
        raise ValueError("verifier timeout must be greater than zero")
    try:
        completed = subprocess.run(
            command,
            cwd=base,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
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
    )


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
        and receipt.tree_sha256 == capture_tree(root).tree_sha256
        and receipt.exit_code == 0
        and receipt.outcome == "passed"
    )
