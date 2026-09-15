"""Explicit, read-only access to pre-Journal workspace records."""

from __future__ import annotations

import json
from pathlib import Path

from .models import WorkspaceRecord, record_from_dict
from .store import WorkspaceStore
from dataclasses import replace


def read_legacy_record(path: str | Path) -> WorkspaceRecord:
    source = Path(path).expanduser().resolve()
    if source.name != "task.json":
        raise ValueError("legacy import requires a task.json path")
    return record_from_dict(json.loads(source.read_text(encoding="utf-8")))


def legacy_requires_import(path: str | Path) -> bool:
    """Legacy records are inspectable but never implicitly resumable."""
    read_legacy_record(path)
    return True


def import_legacy_record(
    path: str | Path, *, store: WorkspaceStore, task_id: str | None = None
) -> WorkspaceRecord:
    """Copy legacy task facts into a new paused Journal-backed task."""
    source = read_legacy_record(path)
    imported = store.create(
        source.task,
        constraints=source.constraints,
        acceptance=source.acceptance,
        task_id=task_id,
    )
    imported = replace(
        imported,
        steps=source.steps,
        decisions=source.decisions,
        artifacts=source.artifacts,
        blockers=(
            *source.blockers,
            "legacy record imported; re-verify before completion",
        ),
        next_action=source.next_action,
        status="paused",
        verifier_results=(),
    )
    store.save(
        imported,
        event_kind="LegacyRecordImported",
        event_payload={
            "source": str(Path(path).expanduser().resolve()),
            "legacy_task_id": source.task_id,
        },
    )
    return imported
