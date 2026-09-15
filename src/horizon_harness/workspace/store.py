"""Atomic file store and single-writer guard for task workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Sequence

from filelock import FileLock

from horizon_harness.messages import Message
from horizon_harness.trace.jsonl import write_jsonl_atomic

from .codec import decode_message, encode_message
from .models import WorkspaceRecord, record_from_dict
from .journal import ExecutionJournal
from .snapshot import capture_tree, write_snapshot


class UnsafeCheckpointError(RuntimeError):
    pass


class WorkspaceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def journal(self, task_id: str) -> ExecutionJournal:
        return ExecutionJournal(self.task_dir(task_id) / "journal.sqlite3")

    def create(
        self,
        task: str,
        *,
        constraints: Sequence[str] = (),
        acceptance: Sequence[str] = (),
        acceptance_contract_id: str = "",
        acceptance_contract_version: int = 0,
        acceptance_commands: Sequence[str] = (),
        task_id: str | None = None,
    ) -> WorkspaceRecord:
        if not task.strip():
            raise ValueError("task must not be empty")
        resolved_id = task_id or uuid.uuid4().hex
        directory = self.task_dir(resolved_id)
        directory.mkdir(parents=True, exist_ok=False)
        record = WorkspaceRecord(
            schema_version=1,
            task_id=resolved_id,
            revision=0,
            task=task,
            constraints=tuple(constraints),
            acceptance=tuple(acceptance),
            acceptance_contract_id=acceptance_contract_id,
            acceptance_contract_version=acceptance_contract_version,
            acceptance_commands=tuple(acceptance_commands),
        )
        self.save(record)
        return record

    def task_dir(self, task_id: str) -> Path:
        if not task_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for ch in task_id
        ):
            raise ValueError(f"invalid task_id {task_id!r}")
        return self.root / task_id

    @contextmanager
    def writer(self, task_id: str) -> Iterator[None]:
        directory = self.task_dir(task_id)
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory / ".writer.lock"), timeout=0):
            yield

    def load(self, task_id: str) -> WorkspaceRecord:
        path = self.task_dir(task_id) / "task.json"
        # Validate the compatibility export even though the SQLite projection
        # is authoritative; corrupted legacy files must fail closed.
        exported = record_from_dict(json.loads(path.read_text(encoding="utf-8")))
        projected = self.journal(task_id).projection()
        if projected is not None:
            return record_from_dict(projected)
        return exported

    def save(
        self,
        record: WorkspaceRecord,
        *,
        event_kind: str = "WorkspaceUpdated",
        event_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = event_payload or {"revision": record.revision}
        self.journal(record.task_id).commit_projection(
            event_kind, payload, record.as_dict()
        )
        self._write_json_atomic(
            self.task_dir(record.task_id) / "task.json", record.as_dict()
        )

    def begin_turn(self, task_id: str, run_id: str, turn: int) -> None:
        self._write_json_atomic(
            self.task_dir(task_id) / "uncommitted.json",
            {"run_id": run_id, "turn": turn},
        )

    def has_uncommitted(self, task_id: str) -> bool:
        return (self.task_dir(task_id) / "uncommitted.json").exists()

    def clear_uncommitted(self, task_id: str) -> None:
        (self.task_dir(task_id) / "uncommitted.json").unlink(missing_ok=True)

    def assert_resumable(self, task_id: str) -> WorkspaceRecord:
        # ``uncommitted.json`` is a legacy turn marker.  A newer journal
        # decision is itself the durable commit point and must take precedence:
        # recovery may continue from its pending ToolBatch without regenerating
        # the model response.  Keep refusing markers that have no complete
        # journal fact, since their side effects cannot be reconciled safely.
        journal = self.journal(task_id)
        pending = journal.recoverable_batches()
        cursor = journal.cursor()
        durable_transition = bool(pending) or cursor.get("phase") == "seal_pending"
        if self.has_uncommitted(task_id) and not durable_transition:
            raise UnsafeCheckpointError(
                "automatic resume refused: the previous process left an uncommitted "
                "turn and tool side effects may be unconfirmed"
            )
        record = self.load(task_id)
        if not record.checkpoint.resumable and not durable_transition:
            raise UnsafeCheckpointError(
                "automatic resume refused: no safe resumable checkpoint"
            )
        return record

    def commit_checkpoint(
        self,
        record: WorkspaceRecord,
        messages: Sequence[Message],
        trace_records: Sequence[dict[str, Any]],
        *,
        code_dir: str | Path | None = None,
    ) -> WorkspaceRecord:
        checkpoint = record.checkpoint
        directory = self.task_dir(record.task_id)
        archive_rel = (
            f"checkpoints/{checkpoint.run_id}-{checkpoint.completed_turn}.json"
        )
        trace_rel = f"traces/{checkpoint.run_id}.jsonl"
        snapshot_rel = f"snapshots/{checkpoint.run_id}-{checkpoint.completed_turn}.json"
        context_rel = f"contexts/{checkpoint.run_id}-{checkpoint.completed_turn}.json"
        prior_refs = checkpoint.message_refs
        if len(prior_refs) > len(messages):
            raise ValueError("checkpoint message references exceed the transcript")
        next_index = sum(ref.startswith(f"{checkpoint.run_id}:") for ref in prior_refs)
        new_refs = tuple(
            f"{checkpoint.run_id}:{index}"
            for index in range(next_index, next_index + len(messages) - len(prior_refs))
        )
        message_refs = (*prior_refs, *new_refs)
        self._write_json_atomic(
            directory / archive_rel,
            {
                "messages": [encode_message(message) for message in messages],
                "message_refs": message_refs,
            },
        )
        write_jsonl_atomic(directory / trace_rel, trace_records)
        snapshot_path = ""
        tree_sha256 = checkpoint.workspace_tree_sha256
        if code_dir is not None:
            snapshot = capture_tree(
                code_dir,
                object_dir=directory / "snapshot-objects",
            )
            write_snapshot(directory / snapshot_rel, snapshot)
            snapshot_path = snapshot_rel
            tree_sha256 = snapshot.tree_sha256
        self._write_json_atomic(
            directory / context_rel,
            {
                "context_manifest_version": 1,
                "task_id": record.task_id,
                "task_revision": record.revision,
                "run_id": checkpoint.run_id,
                "completed_turn": checkpoint.completed_turn,
                "complete_boundary": True,
                "message_refs": message_refs,
                "recent_message_refs": message_refs[-4:],
                "messages_path": archive_rel,
                "snapshot_path": snapshot_path,
                "workspace_tree_sha256": tree_sha256,
                "parent_task_id": record.parent_task_id,
                "parent_revision": record.parent_revision,
                "parent_snapshot_sha256": record.parent_snapshot_sha256,
            },
        )
        committed = replace(
            record,
            checkpoint=replace(
                checkpoint,
                messages_path=archive_rel,
                trace_path=trace_rel,
                message_refs=message_refs,
                recent_message_refs=message_refs[-4:],
                workspace_tree_sha256=tree_sha256,
                snapshot_path=snapshot_path,
                context_manifest_path=context_rel,
            ),
        )
        self.save(
            committed,
            event_kind="BoundaryCommitted",
            event_payload={
                "run_id": checkpoint.run_id,
                "completed_turn": checkpoint.completed_turn,
                "revision": committed.revision,
                "message_refs": message_refs,
            },
        )
        self.clear_uncommitted(record.task_id)
        return committed

    def load_messages(self, record: WorkspaceRecord) -> list[Message]:
        path = self.task_dir(record.task_id) / record.checkpoint.messages_path
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [decode_message(row) for row in raw["messages"]]

    def history_read(self, task_id: str, refs: Sequence[str], max_chars: int) -> str:
        record = self.load(task_id)
        path = self.task_dir(record.task_id) / record.checkpoint.messages_path
        raw = json.loads(path.read_text(encoding="utf-8"))
        messages_by_ref = dict(zip(raw.get("message_refs", ()), raw["messages"]))
        rows: list[str] = []
        remaining = max_chars
        for ref in refs:
            encoded = messages_by_ref.get(ref)
            if encoded is None:
                raise ValueError(f"unknown history reference {ref!r}")
            rendered = json.dumps(encoded, ensure_ascii=False)
            rendered = rendered[:remaining]
            rows.append(f"{ref} {rendered}")
            remaining -= len(rendered)
            if remaining <= 0:
                break
        return "\n".join(rows)

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_json_atomic(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        try:
            with part.open("w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(part, path)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
