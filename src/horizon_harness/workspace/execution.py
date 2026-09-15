"""Adapters that attach durable bookkeeping to controlled tools."""

from __future__ import annotations

import os
import tempfile
import stat
from pathlib import Path
from typing import Any

from horizon_harness.tools import AgentTool, ToolResult, text_result
from horizon_harness.tools import tool_arguments_hash
from horizon_harness.tools.edit import edit_file, edit_fingerprint

from .journal import ExecutionJournal
from .codec import encode_tool_result, decode_tool_result


def atomic_edit_write(path: Path, content: str) -> None:
    """Publish complete bytes, preserving an existing file's permission bits."""
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, name = tempfile.mkstemp(prefix=".hh-edit-", dir=path.parent)
    part = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content.encode("utf-8"))
            stream.flush()
            if hasattr(os, "fchmod"):
                os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(part, path)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        part.unlink(missing_ok=True)


def make_durable_edit_tool(*, root: str | Path, journal: ExecutionJournal) -> AgentTool:
    """Create an edit tool that journals identity and terminal file evidence."""
    base = Path(root).expanduser().resolve()

    def execute(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del on_update
        path = str(args.get("path", ""))
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(base):
            return text_result("Path escapes workspace root.", is_error=True)
        before = (
            resolved.read_text(encoding="utf-8", errors="replace")
            if resolved.is_file()
            else ""
        )
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        replace_all = bool(args.get("replace_all", False))
        if resolved.is_file() and old_string:
            expected_after = before.replace(
                old_string, new_string, -1 if replace_all else 1
            )
        elif not resolved.exists() and old_string == "":
            expected_after = new_string
        else:
            expected_after = before
        input_hash = tool_arguments_hash(args)
        # New plans pass their runtime-owned step identity, not a provider ID.
        # The suffix path remains only for existing direct-tool fixtures.
        step_id = call_id if journal.step(call_id) is not None else f"{call_id}:edit"
        key = f"{step_id}:v1"
        try:
            intent = {
                "path": str(resolved.relative_to(base)),
                "before": edit_fingerprint(before),
                "after": edit_fingerprint(expected_after),
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
            }
            existing = journal.step(step_id)
            if existing is None:
                journal.plan_step(
                    step_id=step_id,
                    tool_name="edit",
                    input_hash=input_hash,
                    idempotency_key=key,
                    effect_class="idempotent_effect",
                    result_payload=intent,
                )
            elif existing["input_hash"] != input_hash:
                return text_result(
                    "existing durable edit step input mismatch", is_error=True
                )
            elif existing["state"] in ("succeeded", "failed"):
                stored = (existing["result_payload"] or {}).get("tool_result")
                if not isinstance(stored, dict):
                    raise ValueError("MISSING_DURABLE_TOOL_RESULT")
                return decode_tool_result(stored)
            elif existing["state"] != "planned":
                return text_result(
                    "edit step is already settled or ambiguous; refusing execution",
                    is_error=True,
                )
            elif existing["result_payload"] is None:
                journal.commit_step_result(
                    step_id, state="planned", result_payload=intent
                )
            elif existing["result_payload"] != intent:
                return text_result(
                    "edit intent already exists; reconciliation required", is_error=True
                )
            else:
                journal.begin_retry(
                    step_id, expected_attempt=existing["attempt"], max_attempts=3
                )
            if abort():
                journal.commit_step_result(
                    step_id, state="cancelled", result_payload={}
                )
                return text_result("Edit aborted before start.", is_error=True)
            result = edit_file(
                path,
                old_string,
                new_string,
                replace_all=replace_all,
                root=base,
                write_content=atomic_edit_write,
            )
            after = (
                resolved.read_text(encoding="utf-8", errors="replace")
                if resolved.is_file()
                else ""
            )
            # env-ok: test-only crash injection
            # env-ok: test-only crash injection
            crash_index = os.environ.get("HORIZON_CRASH_AFTER_EDIT_INDEX")
            if os.environ.get(  # env-ok: test-only crash injection
                "HORIZON_CRASH_AFTER_EDIT"
            ) == "1" or (
                crash_index is not None and step_id.endswith(f":{crash_index}")
            ):
                os._exit(91)
            terminal = "failed" if result.is_error else "succeeded"
            journal.commit_step_result(
                step_id,
                state=terminal,
                result_payload={
                    "path": str(resolved.relative_to(base)),
                    "before": edit_fingerprint(before),
                    "after": edit_fingerprint(after),
                    "tool_result": encode_tool_result(result),
                },
            )
            return result
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    return AgentTool(
        name="edit",
        description="Durable exact file edit with journal bookkeeping.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode="sequential",
    )


def reconcile_edit(
    journal: ExecutionJournal,
    *,
    root: str | Path,
    retry_before: bool = False,
    max_attempts: int = 3,
) -> tuple[str, ...]:
    """Reconcile file evidence; opt-in retries require durable authorization."""
    base = Path(root).expanduser().resolve()
    actions: list[str] = []
    for step_id in journal.planned_step_ids():
        step = journal.step(step_id)
        if step is None:
            continue
        payload = step.get("result_payload") or {}
        # A model decision creates the execution step before the tool starts.
        # Such a step has no effect evidence yet and is safe to execute; only
        # an attempted edit with missing evidence is ambiguous.
        if step["state"] == "planned" and not payload:
            actions.append(f"pending:{step_id}")
            continue
        path_text = payload.get("path")
        before = payload.get("before")
        after = payload.get("after")
        if (
            step["tool_name"] != "edit"
            or not isinstance(path_text, str)
            or not path_text
            or not before
            or not after
        ):
            journal.set_step_state(
                step_id, "ambiguous", {"reason": "missing reconciliation evidence"}
            )
            actions.append(f"ambiguous:{step_id}")
            continue
        relative = Path(path_text)
        path = base / relative
        unsafe_path = relative.is_absolute() or ".." in relative.parts
        for ancestor in (path, *path.parents):
            if ancestor == base:
                break
            if ancestor.is_symlink():
                unsafe_path = True
        if unsafe_path or not path.resolve().is_relative_to(base) or path.is_dir():
            journal.set_step_state(
                step_id, "ambiguous", {**payload, "reason": "unsafe edit path"}
            )
            actions.append(f"ambiguous:{step_id}")
            continue
        content = (
            path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        )
        current = edit_fingerprint(content)
        if current == after:
            from horizon_harness.tools import text_result
            from .codec import encode_tool_result

            reconciled = text_result(f"Edit already applied: {path_text}")
            journal.set_step_state(
                step_id,
                "succeeded",
                {
                    **payload,
                    "reconciled": "after",
                    "tool_result": encode_tool_result(reconciled),
                },
            )
            actions.append(f"succeeded:{step_id}")
        elif current != before:
            journal.set_step_state(
                step_id, "ambiguous", {**payload, "reason": "workspace drift"}
            )
            actions.append(f"ambiguous:{step_id}")
        else:
            if retry_before:
                old = payload.get("old_string")
                new = payload.get("new_string")
                replace_all = payload.get("replace_all")
                valid = (
                    isinstance(old, str)
                    and isinstance(new, str)
                    and isinstance(replace_all, bool)
                )
                if valid:
                    predicted = (
                        content.replace(old, new, -1 if replace_all else 1)
                        if old
                        else new
                    )
                    valid = edit_fingerprint(predicted) == after
                if not valid:
                    journal.set_step_state(
                        step_id,
                        "ambiguous",
                        {**payload, "reason": "invalid retry arguments"},
                    )
                    actions.append(f"ambiguous:{step_id}")
                    continue
                journal.begin_retry(
                    step_id, expected_attempt=step["attempt"], max_attempts=max_attempts
                )
                result = edit_file(
                    str(path),
                    old if isinstance(old, str) else "",
                    new if isinstance(new, str) else "",
                    replace_all=replace_all if isinstance(replace_all, bool) else False,
                    root=base,
                    write_content=atomic_edit_write,
                )
                if result.is_error:
                    journal.set_step_state(
                        step_id, "failed", {**payload, "retry_error": True}
                    )
                    actions.append(f"failed:{step_id}")
                elif edit_fingerprint(path.read_text(encoding="utf-8")) != after:
                    journal.set_step_state(
                        step_id,
                        "ambiguous",
                        {**payload, "reason": "retry output mismatch"},
                    )
                    actions.append(f"ambiguous:{step_id}")
                else:
                    journal.set_step_state(
                        step_id, "succeeded", {**payload, "reconciled": "retry"}
                    )
                    actions.append(f"succeeded:{step_id}")
            else:
                actions.append(f"retry:{step_id}")
    return tuple(actions)
