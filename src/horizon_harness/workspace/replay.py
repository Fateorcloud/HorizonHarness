"""Read-only replay of committed Journal facts."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import shutil
from pathlib import Path
from typing import Any
from horizon_harness.messages import ToolResultBlock

from .journal import ExecutionJournal
from .codec import decode_tool_result


@dataclass(frozen=True)
class ReplayResult:
    events: tuple[tuple[int, str, dict[str, Any]], ...]
    projection: dict[str, Any] | None
    messages: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class RunDiff:
    left_status: str
    right_status: str
    left_tree: str
    right_tree: str
    event_count_delta: dict[str, int]
    message_history_equal: bool


def diff_runs(left: ExecutionJournal, right: ExecutionJournal) -> RunDiff:
    """Compare two runs from their committed replay projections."""
    left_run = replay(left)
    right_run = replay(right)
    left_projection = left_run.projection or {}
    right_projection = right_run.projection or {}
    left_checkpoint = left_projection.get("checkpoint") or {}
    right_checkpoint = right_projection.get("checkpoint") or {}
    left_counts = Counter(kind for _, kind, _ in left_run.events)
    right_counts = Counter(kind for _, kind, _ in right_run.events)
    kinds = sorted(left_counts.keys() | right_counts.keys())
    return RunDiff(
        left_status=str(left_projection.get("status", "")),
        right_status=str(right_projection.get("status", "")),
        left_tree=str(left_checkpoint.get("workspace_tree_sha256", "")),
        right_tree=str(right_checkpoint.get("workspace_tree_sha256", "")),
        event_count_delta={
            kind: right_counts[kind] - left_counts[kind]
            for kind in kinds
            if right_counts[kind] != left_counts[kind]
        },
        message_history_equal=left_run.messages == right_run.messages,
    )


def replay(journal: ExecutionJournal) -> ReplayResult:
    """Read committed facts only; this function has no write or execution path."""
    events = tuple(journal.events())
    messages = tuple(
        payload["message"]
        for _, kind, payload in events
        if kind == "MessageEvent" and isinstance(payload.get("message"), dict)
    )
    return ReplayResult(
        events=events, projection=journal.rebuild_task_projection(), messages=messages
    )


def replay_tool_result_blocks(
    journal: ExecutionJournal, batch_id: str
) -> tuple[ToolResultBlock, ...]:
    """Convert recorded batch results to typed blocks without executing tools."""
    blocks = []
    for row in journal.rebuild_batch_bundle(batch_id):
        encoded = row["result"].get("tool_result")
        if not isinstance(encoded, dict):
            raise ValueError("MISSING_DURABLE_TOOL_RESULT")
        result = decode_tool_result(encoded)
        blocks.append(
            ToolResultBlock(
                tool_call_id=row["call_id"],
                tool_name=row["tool_name"],
                content=result.content,
                is_error=result.is_error,
            )
        )
    return tuple(blocks)


def fork_workspace(source: str | Path, target: str | Path) -> tuple[str, str]:
    """Copy a complete local file boundary into a new empty directory."""
    from .snapshot import capture_tree

    src = Path(source).expanduser().resolve()
    dst = Path(target).expanduser().resolve()
    if not src.is_dir():
        raise ValueError(f"fork source must be a directory: {source}")
    if dst.exists() and any(dst.iterdir()):
        raise ValueError("fork target must be empty")
    if dst == src or dst.is_relative_to(src):
        raise ValueError("fork target must be outside the source workspace")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target_item = dst / item.name
        if item.is_symlink():
            raise ValueError(f"fork does not support symlink entries: {item}")
        if item.is_dir():
            shutil.copytree(item, target_item)
        else:
            shutil.copy2(item, target_item)
    return capture_tree(src).tree_sha256, capture_tree(dst).tree_sha256


def fork_from_checkpoint(
    task_dir: str | Path,
    snapshot_path: str,
    target: str | Path,
    *,
    child_journal: ExecutionJournal | None = None,
    parent_task_id: str = "",
) -> tuple[str, str]:
    from .snapshot import load_snapshot, materialize_snapshot

    base = Path(task_dir).expanduser().resolve()
    manifest = (base / snapshot_path).resolve()
    if not manifest.is_relative_to(base) or not manifest.is_file():
        raise ValueError(
            "checkpoint snapshot manifest is missing or escapes task store"
        )
    snapshot = load_snapshot(manifest)
    materialize_snapshot(snapshot, object_dir=base / "snapshot-objects", target=target)
    if child_journal is not None:
        child_journal.append(
            "ForkCreated",
            {
                "debug_only": True,
                "parent_task_id": parent_task_id,
                "snapshot_path": snapshot_path,
                "source_tree_sha256": snapshot.tree_sha256,
            },
        )
    return snapshot.tree_sha256, snapshot.tree_sha256
