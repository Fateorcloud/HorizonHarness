"""Ordinary agent tools for updating and reading a task workspace."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from horizon_harness.tools import AgentTool, text_result

from .models import Artifact, Decision, TaskStep, WorkspaceRecord
from .store import WorkspaceStore


def make_workspace_update_tool(
    *,
    current: Callable[[], WorkspaceRecord],
    update: Callable[[WorkspaceRecord], None],
    code_dir: Path,
) -> AgentTool:
    def execute(call_id: str, args: dict[str, Any], abort: Any, on_update: Any):
        del call_id, abort, on_update
        try:
            record = current()
            revised = apply_workspace_update(record, args, code_dir=code_dir)
            update(revised)
            return text_result(
                json.dumps(
                    {"task_id": revised.task_id, "revision": revised.revision},
                    sort_keys=True,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            return text_result(str(exc), is_error=True)

    return AgentTool(
        name="workspace_update",
        description=(
            "Update explicit task steps, decisions, artifacts, blockers, next action, "
            "or status. Use the current expected_revision. Artifacts are fingerprinted "
            "by the runtime; verifier results cannot be self-declared here."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expected_revision": {"type": "integer"},
                "steps": {"type": "array", "items": {"type": "object"}},
                "decisions": {"type": "array", "items": {"type": "object"}},
                "artifacts": {"type": "array", "items": {"type": "object"}},
                "blockers": {"type": "array", "items": {"type": "string"}},
                "next_action": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["active", "paused", "completed", "blocked", "failed"],
                },
            },
            "required": ["expected_revision"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode="sequential",
    )


def apply_workspace_update(
    record: WorkspaceRecord, args: dict[str, Any], *, code_dir: Path
) -> WorkspaceRecord:
    """Validate and apply the same update used by the model-facing tool."""

    expected = int(args["expected_revision"])
    if expected != record.revision:
        raise ValueError(
            f"revision conflict: expected {expected}, current {record.revision}"
        )
    step_by_id = {step.id: step for step in record.steps}
    for row in args.get("steps", ()):
        step = TaskStep(
            id=str(row["id"]),
            description=str(row["description"]),
            status=row.get("status", "pending"),
            evidence=tuple(map(str, row.get("evidence", ()))),
        )
        step_by_id[step.id] = step
    step_ids = set(step_by_id)
    decisions = list(record.decisions)
    for row in args.get("decisions", ()):
        decision = Decision(
            content=str(row["content"]),
            rationale=str(row["rationale"]),
            step_ids=tuple(map(str, row.get("step_ids", ()))),
            evidence=tuple(map(str, row.get("evidence", ()))),
        )
        _require_step_refs(decision.step_ids, step_ids)
        decisions.append(decision)
    artifacts = list(record.artifacts)
    for row in args.get("artifacts", ()):
        relative = _safe_relative_path(str(row["path"]))
        path = (code_dir / relative).resolve()
        if not path.is_relative_to(code_dir.resolve()) or not path.is_file():
            raise ValueError(
                f"artifact does not exist inside code directory: {relative}"
            )
        artifact = Artifact(
            path=relative.as_posix(),
            sha256=WorkspaceStore.sha256(path),
            step_ids=tuple(map(str, row.get("step_ids", ()))),
            verification_refs=tuple(map(str, row.get("verification_refs", ()))),
        )
        _require_step_refs(artifact.step_ids, step_ids)
        artifacts = [item for item in artifacts if item.path != artifact.path]
        artifacts.append(artifact)
    requested_status = args.get("status", record.status)
    if requested_status == "completed":
        raise ValueError(
            "workspace_update cannot mark a task completed; a trusted verifier "
            "must pass the completion gate"
        )
    return replace(
        record,
        revision=record.revision + 1,
        steps=tuple(step_by_id.values()),
        decisions=tuple(decisions),
        artifacts=tuple(artifacts),
        blockers=tuple(map(str, args.get("blockers", record.blockers))),
        next_action=str(args.get("next_action", record.next_action)),
        status=requested_status,
    )


def make_history_read_tool(store: WorkspaceStore, task_id: str) -> AgentTool:
    def execute(call_id: str, args: dict[str, Any], abort: Any, on_update: Any):
        del call_id, abort, on_update
        try:
            refs = args.get("refs", ())
            if not isinstance(refs, list) or len(refs) > 20:
                raise ValueError("refs must be a list containing at most 20 references")
            return text_result(store.history_read(task_id, refs, max_chars=8_000))
        except (OSError, ValueError) as exc:
            return text_result(str(exc), is_error=True)

    return AgentTool(
        name="history_read",
        description="Read archived task messages by stable run:index references.",
        parameters={
            "type": "object",
            "properties": {"refs": {"type": "array", "items": {"type": "string"}}},
            "required": ["refs"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode="sequential",
    )


def make_workspace_read_tool(store: WorkspaceStore, task_id: str) -> AgentTool:
    def execute(call_id, args, abort, on_update):
        del call_id, abort, on_update
        try:
            section = args.get("section", "steps")
            if section not in (
                "steps",
                "decisions",
                "artifacts",
                "blockers",
                "verifier_results",
            ):
                raise ValueError("unsupported workspace section")
            offset = int(args.get("offset", 0))
            if offset < 0:
                raise ValueError("offset must be nonnegative")
            record = store.load(task_id)
            items = record.as_dict()[section]
            return text_result(
                json.dumps(
                    {
                        "revision": record.revision,
                        "section": section,
                        "items": items[offset : offset + 3],
                        "next_offset": offset + 3 if offset + 3 < len(items) else None,
                    },
                    ensure_ascii=False,
                )
            )
        except (ValueError, TypeError) as exc:
            return text_result(str(exc), is_error=True)

    return AgentTool(
        name="workspace_read",
        description="Read current persisted task details by section and offset; verify conflicts with historical summary.",
        parameters={
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": [
                        "steps",
                        "decisions",
                        "artifacts",
                        "blockers",
                        "verifier_results",
                    ],
                },
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": ["section"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _safe_relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if path.is_absolute() or not raw or ".." in path.parts:
        raise ValueError(
            f"artifact path must be relative and remain in the code directory: {raw!r}"
        )
    return path


def _require_step_refs(refs: tuple[str, ...], valid: set[str]) -> None:
    missing = [ref for ref in refs if ref not in valid]
    if missing:
        raise ValueError(f"unknown step reference {missing[0]!r}")
