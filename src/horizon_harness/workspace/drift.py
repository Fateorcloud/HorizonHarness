"""Read-only drift evidence and explicit human adoption of external content."""

from __future__ import annotations

import difflib
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, TypeGuard

from .journal import ExecutionJournal


def store_before_image(journal: ExecutionJournal, content: bytes) -> str:
    """Publish a content-addressed object before the edit intent is committed."""
    digest = hashlib.sha256(content).hexdigest()
    objects = journal.path.parent / "snapshot-objects"
    objects.mkdir(parents=True, exist_ok=True)
    target = objects / digest
    if target.exists():
        return digest
    fd, name = tempfile.mkstemp(prefix=".hh-before-", dir=objects)
    part = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, target)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(objects, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        part.unlink(missing_ok=True)
    return digest


def drift_intent(journal: ExecutionJournal, step_id: str) -> dict[str, Any]:
    step = journal.step(step_id)
    payload = (step or {}).get("result_payload") or {}
    if (
        step is None
        or step["tool_name"] != "edit"
        or step["effect_class"] != "idempotent_effect"
        or step["state"] != "ambiguous"
        or payload.get("reason") != "workspace drift"
    ):
        raise ValueError(
            f"step {step_id!r} must be an ambiguous controlled edit with workspace drift"
        )
    return payload


def drifted_step_ids(journal: ExecutionJournal) -> tuple[str, ...]:
    return tuple(
        step_id
        for step_id in journal.unsettled_step_ids()
        if (step := journal.step(step_id))
        and step["tool_name"] == "edit"
        and step["effect_class"] == "idempotent_effect"
        and step["state"] == "ambiguous"
        and (step["result_payload"] or {}).get("reason") == "workspace drift"
    )


def _edit_path(root: str | Path, payload: dict[str, Any]) -> Path:
    base = Path(root).expanduser().resolve()
    path_text = payload.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise ValueError(f"invalid edit path: {path_text!r}")
    relative = Path(path_text)
    path = base / relative
    unsafe = relative.is_absolute() or ".." in relative.parts
    for ancestor in (path, *path.parents):
        if ancestor == base:
            break
        unsafe = unsafe or ancestor.is_symlink()
    if (
        unsafe
        or not path.resolve().is_relative_to(base)
        or (path.exists() and not path.is_file())
    ):
        raise ValueError(f"unsafe edit path: {path_text!r}")
    return path


def _sha256(value: Any) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _text(content: bytes) -> str | None:
    if any(byte < 32 and byte not in b"\t\n\r" for byte in content):
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _diff(expected: str, current: str, path: str, label: str) -> str:
    lines = difflib.unified_diff(
        expected.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile=f"{path} ({label})",
        tofile=f"{path} (current)",
    )
    # difflib does not add the conventional marker for a missing final newline.
    return "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        for line in lines
    )


def drift_report(
    journal: ExecutionJournal, *, root: str | Path, step_id: str
) -> dict[str, Any]:
    """Describe one ambiguous edit without changing its state or workspace."""
    payload = drift_intent(journal, step_id)
    path = _edit_path(root, payload)
    exists = path.is_file()
    current = path.read_bytes() if exists else b""
    old, new = payload.get("old_string"), payload.get("new_string")
    report = {
        "step_id": step_id,
        "path": payload["path"],
        "before_sha256": payload.get("before"),
        "after_sha256": payload.get("after"),
        "current_sha256": hashlib.sha256(current).hexdigest(),
        "before_exists": payload.get("before_exists"),
        "after_exists": payload.get("after_exists"),
        "current_exists": exists,
        "before_image_available": False,
        "binary": _text(current) is None,
        "before_to_current_diff": None,
        "after_to_current_diff": None,
        "old_string_present": isinstance(old, str) and old.encode("utf-8") in current,
        "new_string_present": isinstance(new, str) and new.encode("utf-8") in current,
    }
    digest = payload.get("before")
    if not _sha256(digest):
        report["diff_unavailable_reason"] = "invalid_before_hash"
        return report
    try:
        before = (journal.path.parent / "snapshot-objects" / digest).read_bytes()
    except OSError:
        report["diff_unavailable_reason"] = "before_image_missing"
        return report
    if hashlib.sha256(before).hexdigest() != digest:
        report["diff_unavailable_reason"] = "before_image_hash_mismatch"
        return report
    report["before_image_available"] = True
    before_text, current_text = _text(before), _text(current)
    if before_text is None or current_text is None:
        report["binary"] = True
        report["diff_unavailable_reason"] = "binary"
        return report
    report["before_to_current_diff"] = _diff(
        before_text, current_text, payload["path"], "before"
    )
    replace_all = payload.get("replace_all")
    if (
        not isinstance(old, str)
        or not isinstance(new, str)
        or not isinstance(replace_all, bool)
    ):
        report["after_diff_unavailable_reason"] = "invalid_edit_arguments"
        return report
    after = before_text.replace(old, new, -1 if replace_all else 1) if old else new
    if hashlib.sha256(after.encode("utf-8")).hexdigest() != payload.get("after"):
        report["after_diff_unavailable_reason"] = "after_hash_mismatch"
    elif _text(after.encode("utf-8")) is None:
        report["binary"] = True
        report["after_diff_unavailable_reason"] = "binary"
    else:
        report["after_to_current_diff"] = _diff(
            after, current_text, payload["path"], "expected after"
        )
    return report


def adopt_current(
    journal: ExecutionJournal,
    *,
    root: str | Path,
    step_id: str,
    expected_sha: str,
    note: str = "",
) -> dict[str, Any]:
    """Check the operator's observation, then atomically cancel journal work."""
    if not _sha256(expected_sha):
        raise ValueError(
            f"expected-sha must be a lowercase SHA-256 digest: {expected_sha!r}"
        )
    payload = drift_intent(journal, step_id)
    path = _edit_path(root, payload)
    current = hashlib.sha256(path.read_bytes() if path.is_file() else b"").hexdigest()
    if current != expected_sha:
        raise ValueError(
            f"expected-sha mismatch for {payload['path']}: expected {expected_sha}, current {current}"
        )
    cancelled = journal.cancel_drift_edit(
        step_id, expected_payload=payload, sha256=current, note=note
    )
    return {
        "step_id": step_id,
        "path": payload["path"],
        "sha256": current,
        "cancelled_steps": cancelled,
        "note": note,
    }


def drift_resume_message(task_id: str, step_id: str) -> str:
    return (
        "automatic resume refused: controlled edit has ambiguous workspace drift; "
        f"task_id={task_id}, step_id={step_id}. "
        f"Use inspect {task_id} to review, then adopt-current {task_id} "
        f"--step {step_id} --expected-sha <current_sha256> to adopt current content."
    )
