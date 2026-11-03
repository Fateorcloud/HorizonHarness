"""Small SQLite execution journal used by durable task runs.

The journal is deliberately boring: append-only facts with a monotonic sequence
number, and a projection cursor.  It is an audit/rebuild seam, not a second
agent loop.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator
from horizon_harness.messages import AssistantMessage, tool_results_of
from .codec import encode_message, decode_message


class ExecutionJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5.0)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS journal (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS projection (
                name TEXT PRIMARY KEY,
                applied_seq INTEGER NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS task_projection (
                name TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS execution_steps (
                step_id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                effect_class TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt INTEGER NOT NULL CHECK (attempt > 0),
                result_payload TEXT
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS execution_cursor (
                name TEXT PRIMARY KEY,
                turn_no INTEGER NOT NULL,
                phase TEXT NOT NULL,
                next_tool_index INTEGER NOT NULL,
                batch_id TEXT
            )""")
            columns = {
                row[1]
                for row in db.execute("PRAGMA table_info(execution_cursor)").fetchall()
            }
            if "batch_id" not in columns:
                db.execute("ALTER TABLE execution_cursor ADD COLUMN batch_id TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS tool_batches (
                batch_id TEXT PRIMARY KEY,
                turn_no INTEGER NOT NULL,
                calls_json TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS tool_batch_results (
                batch_id TEXT NOT NULL,
                call_index INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, call_index)
            )""")
            batch_columns = {
                row[1] for row in db.execute("PRAGMA table_info(tool_batches)")
            }
            if "next_tool_index" in batch_columns:
                legacy = db.execute(
                    "SELECT batch_id, turn_no, calls_json, next_tool_index, "
                    "bundle_committed FROM tool_batches"
                ).fetchall()
                for batch_id, turn_no, _, next_index, committed in legacy:
                    result_count = int(
                        db.execute(
                            "SELECT count(*) FROM tool_batch_results WHERE batch_id=?",
                            (batch_id,),
                        ).fetchone()[0]
                    )
                    if int(next_index) > result_count:
                        raise ValueError(
                            "LEGACY_MUTABLE_BATCH_PROGRESS: explicit import required"
                        )
                    if committed and not self._bundle_commit_exists(db, batch_id):
                        raise ValueError(
                            "LEGACY_UNSCOPED_BUNDLE_COMMIT: explicit import required"
                        )
                db.execute(
                    "CREATE TABLE tool_batches_immutable ("
                    "batch_id TEXT PRIMARY KEY, turn_no INTEGER NOT NULL, "
                    "calls_json TEXT NOT NULL)"
                )
                db.execute(
                    "INSERT INTO tool_batches_immutable(batch_id, turn_no, calls_json) "
                    "SELECT batch_id, turn_no, calls_json FROM tool_batches"
                )
                db.execute("DROP TABLE tool_batches")
                db.execute("ALTER TABLE tool_batches_immutable RENAME TO tool_batches")
            db.execute(
                "INSERT OR IGNORE INTO projection(name, applied_seq) VALUES ('task', 0)"
            )
            db.execute(
                "INSERT OR IGNORE INTO execution_cursor(name, turn_no, phase, next_tool_index, batch_id) VALUES ('task', 0, 'ready', 0, NULL)"
            )

    def set_cursor(
        self,
        *,
        turn_no: int,
        phase: str,
        next_tool_index: int = 0,
        batch_id: str | None = None,
    ) -> int:
        if turn_no < 0 or next_tool_index < 0 or not phase.strip():
            raise ValueError("invalid execution cursor")
        with self._connect() as db:
            self._append_step_event(
                db,
                "CursorAdvanced",
                {
                    "turn_no": turn_no,
                    "phase": phase,
                    "next_tool_index": next_tool_index,
                },
            )
            db.execute(
                "UPDATE execution_cursor SET turn_no=?, phase=?, next_tool_index=?, batch_id=? WHERE name='task'",
                (turn_no, phase, next_tool_index, batch_id),
            )
            return turn_no

    def cursor(self) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT turn_no, phase, next_tool_index, batch_id FROM execution_cursor WHERE name='task'"
            ).fetchone()
        result = {
            "turn_no": int(row[0]),
            "phase": row[1],
            "next_tool_index": int(row[2]),
        }
        if row[3] is not None:
            result["batch_id"] = row[3]
        return result

    def plan_batch(
        self,
        *,
        batch_id: str,
        turn_no: int,
        calls: list[dict[str, Any]],
        assistant_message: dict[str, Any] | None = None,
    ) -> None:
        if not batch_id or turn_no < 1 or not calls:
            raise ValueError("invalid tool batch")
        with self._connect() as db:
            db.execute(
                "INSERT INTO tool_batches(batch_id, turn_no, calls_json) VALUES (?, ?, ?)",
                (
                    batch_id,
                    turn_no,
                    json.dumps(
                        {"calls": calls, "assistant_message": assistant_message},
                        sort_keys=True,
                    ),
                ),
            )

    def commit_model_decision(
        self,
        *,
        events: list[tuple[str, dict[str, Any]]],
        batch_id: str | None,
        turn_no: int,
        calls: list[dict[str, Any]],
        assistant_message: dict[str, Any],
    ) -> int:
        """Atomically publish the assistant decision and its durable batch plan."""
        decoded = decode_message(assistant_message)
        if decoded.role != "assistant":
            raise ValueError("model decision requires an AssistantMessage")
        assistant_message = encode_message(decoded)
        if not isinstance(decoded, AssistantMessage):
            raise ValueError("model decision requires an AssistantMessage")
        actual_calls = [
            {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
            for call in decoded.tool_calls
        ]
        if actual_calls != [
            {key: call.get(key) for key in ("id", "name", "arguments")}
            for call in calls
        ]:
            raise ValueError("ToolBatch does not match canonical AssistantMessage")
        if bool(calls) != bool(batch_id):
            raise ValueError("tool calls require exactly one batch identity")
        normalized = []
        message_count = 0
        for kind, payload in events:
            if kind == "MessageEvent":
                message_count += 1
                if (
                    encode_message(decode_message(payload["message"]))
                    != assistant_message
                ):
                    raise ValueError(
                        "decision MessageEvent differs from AssistantMessage"
                    )
                payload = {**payload, "message": assistant_message}
            normalized.append((kind, payload))
        if message_count != 1:
            raise ValueError("model decision requires exactly one MessageEvent")
        with self._connect() as db:
            for kind, payload in normalized:
                db.execute(
                    "INSERT INTO journal(event_id, kind, payload) VALUES (?, ?, ?)",
                    (str(uuid.uuid4()), kind, json.dumps(payload, sort_keys=True)),
                )
            if batch_id is not None:
                self._append_step_event(
                    db,
                    "ToolBatchPlan",
                    {
                        "batch_id": batch_id,
                        "turn_no": turn_no,
                        "calls": calls,
                        "assistant_message": assistant_message,
                    },
                )
                db.execute(
                    "INSERT INTO tool_batches(batch_id, turn_no, calls_json) VALUES (?, ?, ?)",
                    (
                        batch_id,
                        turn_no,
                        json.dumps(
                            {"calls": calls, "assistant_message": assistant_message},
                            sort_keys=True,
                        ),
                    ),
                )
                for index, call in enumerate(calls):
                    step_id = call.get("execution_step_id", f"{batch_id}:{index}")
                    self._append_step_event(
                        db,
                        "StepPlanned",
                        {
                            "step_id": step_id,
                            "tool_name": call["name"],
                            "input_hash": call.get("args_hash", ""),
                            "idempotency_key": f"{step_id}:v1",
                            "effect_class": "idempotent_effect",
                            "state": "planned",
                            "attempt": 1,
                            "result_payload": None,
                        },
                    )
                    db.execute(
                        "INSERT INTO execution_steps(step_id, tool_name, input_hash, idempotency_key, effect_class, state, attempt, result_payload) VALUES (?, ?, ?, ?, 'idempotent_effect', 'planned', 1, NULL)",
                        (
                            step_id,
                            call["name"],
                            call.get("args_hash", ""),
                            f"{step_id}:v1",
                        ),
                    )
            seq = int(db.execute("SELECT max(seq) FROM journal").fetchone()[0])
            db.execute(
                "UPDATE execution_cursor SET turn_no=?, phase=?, next_tool_index=?, batch_id=? WHERE name='task'",
                (turn_no, "tools_pending" if batch_id else "seal_pending", 0, batch_id),
            )
            db.execute("UPDATE projection SET applied_seq=? WHERE name='task'", (seq,))
            return seq

    def batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT batch_id, turn_no, calls_json FROM tool_batches WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
        if row is None:
            return None
        encoded = json.loads(row[2])
        if isinstance(encoded, list):
            encoded = {"calls": encoded, "assistant_message": None}
        if not isinstance(encoded, dict):
            raise ValueError(f"invalid tool batch plan: {batch_id}")
        raw_calls = encoded.get("calls")
        if not isinstance(raw_calls, list) or not all(
            isinstance(item, dict) for item in raw_calls
        ):
            raise ValueError(f"invalid tool batch plan: {batch_id}")
        calls = raw_calls
        return {
            "batch_id": row[0],
            "turn_no": row[1],
            "calls": calls,
            "assistant_message": encoded.get("assistant_message"),
            "next_tool_index": self._committed_prefix(row[0], calls),
            "bundle_committed": self._batch_bundle_committed(row[0]),
        }

    def batch_assistant_message(self, batch_id: str):
        """Decode the exact canonical assistant message persisted in a batch."""
        batch = self.batch(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        value = batch.get("assistant_message")
        if not isinstance(value, dict):
            raise ValueError("MISSING_DURABLE_ASSISTANT_MESSAGE")
        from .codec import decode_message

        return decode_message(value)

    def rebuild_messages(self):
        """Decode committed canonical messages in journal order."""
        from .codec import decode_message

        messages = []
        for seq, kind, payload in self.events():
            if kind != "MessageEvent":
                continue
            value = payload.get("message")
            if not isinstance(value, dict):
                raise ValueError(f"message event {seq} is missing canonical payload")
            messages.append(decode_message(value))
        return tuple(messages)

    def recoverable_batch(self, batch_id: str) -> dict[str, Any] | None:
        batch = self.batch(batch_id)
        if batch is None or batch["bundle_committed"]:
            return None
        # The plan is immutable. Derive the suffix from durable result facts,
        # rather than trusting the legacy mutable progress column. Results must
        # form a contiguous prefix of the original ordered calls.
        result_rows = self.batch_results(batch_id)
        committed = {row.get("call_id") for row in result_rows}
        for call in batch["calls"]:
            step_id = call.get("execution_step_id")
            step = self.step(step_id) if isinstance(step_id, str) else None
            payload = step.get("result_payload") if step else None
            if (
                step
                and step.get("state") in ("succeeded", "failed", "cancelled")
                and isinstance(payload, dict)
            ):
                if isinstance(payload.get("tool_result"), dict):
                    committed.add(call.get("id"))
        index = 0
        while (
            index < len(batch["calls"]) and batch["calls"][index].get("id") in committed
        ):
            index += 1
        if any(call.get("id") in committed for call in batch["calls"][index:]):
            raise ValueError("tool batch results are not an ordered prefix")
        batch["remaining_calls"] = tuple(batch["calls"][index:])
        return batch

    def recoverable_batches(self) -> tuple[dict[str, Any], ...]:
        with self._connect() as db:
            ids = [
                row[0]
                for row in db.execute(
                    "SELECT batch_id FROM tool_batches ORDER BY turn_no, rowid"
                ).fetchall()
            ]
        batches = []
        for batch_id in ids:
            batch = self.recoverable_batch(batch_id)
            if batch is not None:
                batches.append(batch)
        return tuple(batches)

    def commit_batch_bundle(self, batch_id: str, message_event: dict[str, Any]) -> bool:
        """Atomically commit one batch's result bundle and cursor transition."""
        message_payload = message_event.get("message")
        if not isinstance(message_payload, dict):
            raise ValueError("batch bundle requires a canonical MessageEvent")
        message = decode_message(message_payload)
        if message.kind != "tool_result":
            raise ValueError("batch bundle requires a tool-result message")
        with self._connect() as db:
            row = db.execute(
                "SELECT turn_no, calls_json FROM tool_batches WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
            if row is None:
                raise KeyError(batch_id)
            if self._bundle_commit_exists(db, batch_id):
                return False
            encoded = json.loads(row[1])
            calls = encoded if isinstance(encoded, list) else encoded["calls"]
            expected_ids = tuple(call.get("id") for call in calls)
            actual_ids = tuple(
                block.tool_call_id for block in tool_results_of(message.content)
            )
            if actual_ids != expected_ids:
                raise ValueError("batch bundle results do not match the batch plan")
            result_count = int(
                db.execute(
                    "SELECT count(*) FROM tool_batch_results WHERE batch_id=?",
                    (batch_id,),
                ).fetchone()[0]
            )
            if result_count != len(calls):
                raise ValueError("batch bundle requires all durable tool results")
            event_payload = {**message_event, "batch_id": batch_id}
            cursor = db.execute(
                "INSERT INTO journal(event_id, kind, payload) VALUES (?, 'MessageEvent', ?)",
                (
                    str(uuid.uuid4()),
                    json.dumps(event_payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            self._append_step_event(
                db,
                "ToolBatchBundleCommitted",
                {"batch_id": batch_id, "turn_no": int(row[0])},
            )
            self._append_step_event(
                db,
                "CursorAdvanced",
                {
                    "turn_no": int(row[0]),
                    "phase": "seal_pending",
                    "next_tool_index": len(calls),
                    "batch_id": None,
                },
            )
            db.execute(
                "UPDATE execution_cursor SET turn_no=?, phase='seal_pending', "
                "next_tool_index=?, batch_id=NULL WHERE name='task'",
                (int(row[0]), len(calls)),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("bundle insert did not return a sequence")
            return True

    def record_batch_result(
        self,
        batch_id: str,
        call_index: int,
        result: dict[str, Any],
    ) -> None:
        if call_index < 0:
            raise ValueError("call_index must be nonnegative")
        with self._connect() as db:
            if self._bundle_commit_exists(db, batch_id):
                raise ValueError("batch already committed")
            committed_indices = tuple(
                int(row[0])
                for row in db.execute(
                    "SELECT call_index FROM tool_batch_results WHERE batch_id=? "
                    "ORDER BY call_index",
                    (batch_id,),
                ).fetchall()
            )
            if committed_indices != tuple(range(len(committed_indices))):
                raise ValueError("batch results are not a contiguous prefix")
            if call_index != len(committed_indices):
                raise ValueError("batch result is duplicate or out of order")
            db.execute(
                "INSERT INTO tool_batch_results(batch_id, call_index, result_json) VALUES (?, ?, ?)",
                (
                    batch_id,
                    call_index,
                    json.dumps(result, ensure_ascii=False, sort_keys=True),
                ),
            )

    def cancel_drift_edit(
        self,
        step_id: str,
        *,
        expected_payload: dict[str, Any],
        sha256: str,
        note: str = "",
    ) -> tuple[str, ...]:
        """Commit human adoption, suffix cancellations and results as one fact set."""
        from horizon_harness.tools import text_result
        from .codec import encode_tool_result

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT tool_name, state, effect_class, result_payload FROM execution_steps WHERE step_id=?",
                (step_id,),
            ).fetchone()
            if (
                row is None
                or row[:3] != ("edit", "ambiguous", "idempotent_effect")
                or json.loads(row[3] or "{}") != expected_payload
                or expected_payload.get("reason") != "workspace drift"
            ):
                raise ValueError(
                    f"step {step_id!r} must be an ambiguous controlled edit with workspace drift"
                )
            owners = []
            for batch_id, encoded in db.execute(
                "SELECT batch_id, calls_json FROM tool_batches"
            ):
                plan = json.loads(encoded)
                calls = plan if isinstance(plan, list) else plan["calls"]
                for index, call in enumerate(calls):
                    if call.get("execution_step_id") == step_id:
                        owners.append((batch_id, calls, index))
            if len(owners) != 1:
                raise ValueError(
                    f"drift step must belong to exactly one tool batch: {step_id}"
                )
            batch_id, calls, adopted_index = owners[0]
            if self._bundle_commit_exists(db, batch_id):
                raise ValueError("batch already committed")
            result_rows = db.execute(
                "SELECT call_index FROM tool_batch_results WHERE batch_id=? ORDER BY call_index",
                (batch_id,),
            ).fetchall()
            indices = tuple(row[0] for row in result_rows)
            if indices != tuple(range(len(indices))) or adopted_index < len(indices):
                raise ValueError(
                    "drift adoption requires an uncommitted result in an ordered batch"
                )

            # A start event counts as evidence even if the tool never saved a payload.
            facts = [
                (seq, kind, json.loads(raw))
                for seq, kind, raw in db.execute(
                    "SELECT seq, kind, payload FROM journal WHERE kind IN "
                    "('ToolBatchPlan', 'ToolExecutionStartEvent', 'StepResultCommitted', 'StepRetryStarted')"
                )
            ]
            plan_seq = max(
                (
                    seq
                    for seq, kind, fact in facts
                    if kind == "ToolBatchPlan" and fact.get("batch_id") == batch_id
                ),
                default=0,
            )
            started_calls = {
                fact.get("tool_call_id")
                for seq, kind, fact in facts
                if seq > plan_seq and kind == "ToolExecutionStartEvent"
            }
            evidenced_steps = {
                fact.get("step_id")
                for _, kind, fact in facts
                if kind in ("StepResultCommitted", "StepRetryStarted")
            }
            cancelled = []
            for index, call in enumerate(calls):
                if index < adopted_index:
                    continue
                current_id = call.get("execution_step_id")
                step = db.execute(
                    "SELECT state, attempt, result_payload FROM execution_steps WHERE step_id=?",
                    (current_id,),
                ).fetchone()
                if step is None:
                    raise ValueError(f"missing batch execution step: {current_id}")
                payload = json.loads(step[2]) if step[2] else {}
                if index != adopted_index and (
                    step[0] != "planned"
                    or step[1] != 1
                    or payload
                    or current_id in evidenced_steps
                    or call.get("id") in started_calls
                ):
                    continue
                if index == adopted_index:
                    reason = "adopted_external_content"
                    message = (
                        f"文件 {payload['path']} 在任务中断期间被外部修改，"
                        f"操作人员已采用当前内容（sha256 {sha256}）；"
                        "本次编辑未执行，修改前请先重新读取该文件"
                    )
                    payload.update(reason=reason, sha256=sha256, note=note)
                else:
                    payload.update(
                        reason="cancelled_after_adopted_edit", cancelled_by=step_id
                    )
                    message = "因前序编辑被取消而未执行，请重新规划"
                payload["tool_result"] = encode_tool_result(
                    text_result(message, is_error=True)
                )
                self._append_step_event(
                    db,
                    "StepResultCommitted",
                    {
                        "step_id": current_id,
                        "state": "cancelled",
                        "result_payload": payload,
                    },
                )
                db.execute(
                    "UPDATE execution_steps SET state='cancelled', result_payload=? WHERE step_id=?",
                    (
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        current_id,
                    ),
                )
                cancelled.append(current_id)

            # Preserve the ordered-prefix contract, including completed results
            # whose tool effect committed before a process exit.
            for index in range(len(indices), len(calls)):
                call = calls[index]
                step = db.execute(
                    "SELECT state, result_payload FROM execution_steps WHERE step_id=?",
                    (call.get("execution_step_id"),),
                ).fetchone()
                payload = json.loads(step[1]) if step and step[1] else {}
                stored = payload.get("tool_result")
                if (
                    not step
                    or step[0] not in ("succeeded", "failed", "cancelled")
                    or not isinstance(stored, dict)
                ):
                    break
                result = {
                    "call_id": call["id"],
                    "is_error": stored["is_error"],
                    "terminate": stored["terminate"],
                    "details": stored["details"],
                    "tool_result": stored,
                }
                db.execute(
                    "INSERT INTO tool_batch_results(batch_id, call_index, result_json) VALUES (?, ?, ?)",
                    (
                        batch_id,
                        index,
                        json.dumps(result, ensure_ascii=False, sort_keys=True),
                    ),
                )
                self._append_step_event(
                    db,
                    "ToolBatchResultCommitted",
                    {
                        "batch_id": batch_id,
                        "call_index": index,
                        "result": result,
                    },
                )
            return tuple(cancelled)

    @staticmethod
    def _bundle_commit_exists(db: sqlite3.Connection, batch_id: str) -> bool:
        rows = db.execute(
            "SELECT payload FROM journal WHERE kind='ToolBatchBundleCommitted'"
        ).fetchall()
        return any(json.loads(row[0]).get("batch_id") == batch_id for row in rows)

    def _batch_bundle_committed(self, batch_id: str) -> bool:
        with self._connect() as db:
            return self._bundle_commit_exists(db, batch_id)

    def _committed_prefix(self, batch_id: str, calls: list[dict[str, Any]]) -> int:
        committed = {row.get("call_id") for row in self.batch_results(batch_id)}
        index = 0
        while index < len(calls) and calls[index].get("id") in committed:
            index += 1
        return index

    def batch_results(self, batch_id: str) -> tuple[dict[str, Any], ...]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT result_json FROM tool_batch_results WHERE batch_id=? ORDER BY call_index",
                (batch_id,),
            ).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def materialize_terminal_batch_results(self, batch_id: str) -> None:
        """Project settled execution-step facts into the ordered batch results."""
        batch = self.batch(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        existing_ids = {row.get("call_id") for row in self.batch_results(batch_id)}
        for call_index, call in enumerate(batch["calls"]):
            call_id = call.get("id")
            if call_id in existing_ids:
                continue
            step_id = call.get("execution_step_id")
            step = self.step(step_id) if isinstance(step_id, str) else None
            payload = step.get("result_payload") if step else None
            stored = payload.get("tool_result") if isinstance(payload, dict) else None
            if (
                step
                and step.get("state") in ("succeeded", "failed", "cancelled")
                and isinstance(stored, dict)
            ):
                from .codec import decode_tool_result

                result = decode_tool_result(stored)
                self.record_batch_result(
                    batch_id,
                    call_index,
                    {
                        "call_id": call_id,
                        "is_error": result.is_error,
                        "terminate": result.terminate,
                        "details": result.details,
                        "tool_result": stored,
                    },
                )
                existing_ids.add(call_id)

    def decoded_batch_results(self, batch_id: str):
        """Return committed results keyed by provider call id in plan order."""
        from .codec import decode_tool_result

        decoded = []
        for row in self.batch_results(batch_id):
            call_id = row.get("call_id")
            stored = row.get("tool_result")
            if not isinstance(call_id, str) or not isinstance(stored, dict):
                raise ValueError(f"invalid durable tool result in batch {batch_id}")
            decoded.append((call_id, decode_tool_result(stored)))
        return tuple(decoded)

    def turn_has_terminating_batch(self, turn_no: int) -> bool:
        """Read the terminal flag needed to finish a sealed crash boundary."""
        with self._connect() as db:
            batch_ids = tuple(
                row[0]
                for row in db.execute(
                    "SELECT batch_id FROM tool_batches WHERE turn_no=? ORDER BY rowid",
                    (turn_no,),
                ).fetchall()
            )
        if len(batch_ids) > 1:
            raise ValueError(f"turn {turn_no} has multiple durable tool batches")
        if not batch_ids:
            return False
        return any(
            result.terminate for _, result in self.decoded_batch_results(batch_ids[0])
        )

    def record_batch_tool_result(
        self, batch_id: str, call_index: int, call_id: str, result: Any
    ) -> None:
        from .codec import encode_tool_result

        self.record_batch_result(
            batch_id,
            call_index,
            {
                "call_id": call_id,
                "is_error": result.is_error,
                "terminate": result.terminate,
                "details": result.details,
                "tool_result": encode_tool_result(result),
            },
        )

    def rebuild_batch_bundle(self, batch_id: str) -> tuple[dict[str, Any], ...]:
        batch = self.batch(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        results = self.batch_results(batch_id)
        by_id = {row.get("call_id"): row for row in results}
        bundle = []
        for call in batch["calls"]:
            result = by_id.get(call.get("id"))
            if result is None:
                raise ValueError(f"batch result missing for call: {call.get('id')}")
            bundle.append(
                {"call_id": call["id"], "tool_name": call.get("name"), "result": result}
            )
        return tuple(bundle)

    def plan_step(
        self,
        *,
        step_id: str,
        tool_name: str,
        input_hash: str,
        idempotency_key: str,
        effect_class: str,
        result_payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist one logical tool step before it can execute."""
        values = (step_id, tool_name, input_hash, idempotency_key, effect_class)
        with self._connect() as db:
            self._append_step_event(
                db,
                "StepPlanned",
                {
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "input_hash": input_hash,
                    "idempotency_key": idempotency_key,
                    "effect_class": effect_class,
                    "state": "planned",
                    "attempt": 1,
                    "result_payload": result_payload,
                },
            )
            db.execute(
                "INSERT INTO execution_steps(step_id, tool_name, input_hash, "
                "idempotency_key, effect_class, state, attempt, result_payload) VALUES (?, ?, ?, ?, ?, 'planned', 1, ?)",
                (
                    *values,
                    json.dumps(result_payload, ensure_ascii=False, sort_keys=True)
                    if result_payload is not None
                    else None,
                ),
            )

    def step(self, step_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT step_id, tool_name, input_hash, idempotency_key, effect_class, "
                "state, attempt, result_payload FROM execution_steps WHERE step_id=?",
                (step_id,),
            ).fetchone()
        if row is None:
            return None
        keys = (
            "step_id",
            "tool_name",
            "input_hash",
            "idempotency_key",
            "effect_class",
            "state",
            "attempt",
            "result_payload",
        )
        result = dict(zip(keys, row))
        if result["result_payload"] is not None:
            result["result_payload"] = json.loads(result["result_payload"])
        return result

    def begin_retry(
        self, step_id: str, *, expected_attempt: int, max_attempts: int
    ) -> int:
        """Persist retry authorization before IO; stale callers cannot retry."""
        if expected_attempt < 1 or max_attempts < 1:
            raise ValueError("attempt counts must be positive")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT state, attempt FROM execution_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown execution step: {step_id}")
            if row != ("planned", expected_attempt):
                raise ValueError(
                    f"retry requires planned step at attempt {expected_attempt}: {row!r}"
                )
            if expected_attempt >= max_attempts:
                raise ValueError(f"retry budget exhausted for step {step_id}")
            attempt = expected_attempt + 1
            self._append_step_event(
                db,
                "StepRetryStarted",
                {
                    "step_id": step_id,
                    "attempt": attempt,
                },
            )
            db.execute(
                "UPDATE execution_steps SET attempt=? WHERE step_id=?",
                (attempt, step_id),
            )
            return attempt

    def planned_step_ids(self) -> tuple[str, ...]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT step_id FROM execution_steps WHERE state='planned' ORDER BY rowid"
            ).fetchall()
        return tuple(row[0] for row in rows)

    def commit_step_result(
        self, step_id: str, *, state: str, result_payload: dict[str, Any]
    ) -> None:
        if state not in {"planned", "succeeded", "failed", "ambiguous", "cancelled"}:
            raise ValueError(f"invalid terminal step state: {state!r}")
        with self._connect() as db:
            self._append_step_event(
                db,
                "StepResultCommitted",
                {
                    "step_id": step_id,
                    "state": state,
                    "result_payload": result_payload,
                },
            )
            updated = db.execute(
                "UPDATE execution_steps SET state=?, result_payload=? WHERE step_id=?",
                (
                    state,
                    json.dumps(result_payload, ensure_ascii=False, sort_keys=True),
                    step_id,
                ),
            ).rowcount
            if updated != 1:
                raise KeyError(f"unknown execution step: {step_id}")

    @staticmethod
    def _append_step_event(
        db: sqlite3.Connection, kind: str, payload: dict[str, Any]
    ) -> None:
        cursor = db.execute(
            "INSERT INTO journal(event_id, kind, payload) VALUES (?, ?, ?)",
            (
                str(uuid.uuid4()),
                kind,
                json.dumps({"step_event_version": 1, **payload}, sort_keys=True),
            ),
        )
        db.execute(
            "UPDATE projection SET applied_seq=? WHERE name='task'", (cursor.lastrowid,)
        )

    def rebuild_steps(self) -> dict[str, dict[str, Any]]:
        """Reconstruct recorded step state without consulting execution_steps."""
        steps: dict[str, dict[str, Any]] = {}
        for seq, kind, payload in self.events():
            if kind not in ("StepPlanned", "StepResultCommitted", "StepRetryStarted"):
                continue
            if payload.get("step_event_version") != 1:
                raise ValueError(f"unsupported step event version at {seq}")
            item = {
                key: value
                for key, value in payload.items()
                if key != "step_event_version"
            }
            step_id = item["step_id"]
            if kind == "StepPlanned":
                if step_id in steps:
                    raise ValueError(f"duplicate planned step at {seq}: {step_id}")
                steps[step_id] = item
            else:
                if step_id not in steps:
                    raise ValueError(f"result without planned step at {seq}: {step_id}")
                steps[step_id].update(item)
        return steps

    def set_step_state(
        self, step_id: str, state: str, result_payload: dict[str, Any]
    ) -> None:
        self.commit_step_result(step_id, state=state, result_payload=result_payload)

    def unsettled_step_ids(self) -> tuple[str, ...]:
        """Unknown states are unsafe too; only confirmed settled states pass."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT step_id FROM execution_steps "
                "WHERE state NOT IN ('succeeded', 'failed', 'cancelled') "
                "ORDER BY step_id"
            ).fetchall()
        return tuple(row[0] for row in rows)

    def commit_projection(
        self, kind: str, payload: dict[str, Any], projection: dict[str, Any]
    ) -> int:
        """Atomically append a fact and replace the task projection."""
        if not kind.strip():
            raise ValueError("journal event kind must not be empty")
        if "task_projection" in payload or "projection_version" in payload:
            raise ValueError("projection event payload contains reserved fields")
        event_payload = {
            **payload,
            "projection_version": 1,
            "task_projection": projection,
        }
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO journal(event_id, kind, payload) VALUES (?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    kind,
                    json.dumps(event_payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            if cur.lastrowid is None:
                raise RuntimeError("journal insert did not return a sequence")
            seq = int(cur.lastrowid)
            db.execute(
                "INSERT INTO task_projection(name, payload) VALUES ('task', ?) "
                "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
                (json.dumps(projection, ensure_ascii=False, sort_keys=True),),
            )
            db.execute("UPDATE projection SET applied_seq=? WHERE name='task'", (seq,))
            return seq

    def rebuild_task_projection(self) -> dict[str, Any] | None:
        """Derive the latest recorded task state without reading its cache.

        Early journals lacked state payloads; reject those update events rather
        than presenting an incomplete reconstruction as successful replay.
        """
        result = None
        state_kinds = {
            "WorkspaceUpdated",
            "BoundaryCommitted",
            "VerificationRecorded",
            "CompletionAccepted",
        }
        for seq, kind, payload in self.events():
            if "projection_version" not in payload:
                if kind in state_kinds:
                    raise ValueError(f"event {seq} lacks reconstructable task state")
                continue
            if payload["projection_version"] != 1:
                raise ValueError(f"unsupported projection version at event {seq}")
            result = payload.get("task_projection")
            if not isinstance(result, dict):
                raise ValueError(f"invalid task projection at event {seq}")
        return result

    def projection(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM task_projection WHERE name='task'"
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def append(self, kind: str, payload: dict[str, Any]) -> int:
        if not kind.strip():
            raise ValueError("journal event kind must not be empty")
        with self._connect() as db:
            row = db.execute(
                "INSERT INTO journal(event_id, kind, payload) VALUES (?, ?, ?) RETURNING seq",
                (
                    str(uuid.uuid4()),
                    kind,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            ).fetchone()
            assert row is not None
            seq = int(row[0])
            db.execute("UPDATE projection SET applied_seq=? WHERE name='task'", (seq,))
            return seq

    def append_many(self, events: Iterable[tuple[str, dict[str, Any]]]) -> int:
        rows = list(events)
        if not rows:
            return self.applied_seq()
        with self._connect() as db:
            for kind, payload in rows:
                if not kind.strip():
                    raise ValueError("journal event kind must not be empty")
                db.execute(
                    "INSERT INTO journal(event_id, kind, payload) VALUES (?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        kind,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    ),
                )
            seq = int(db.execute("SELECT max(seq) FROM journal").fetchone()[0])
            db.execute("UPDATE projection SET applied_seq=? WHERE name='task'", (seq,))
            return seq

    def applied_seq(self) -> int:
        with self._connect() as db:
            return int(
                db.execute(
                    "SELECT applied_seq FROM projection WHERE name='task'"
                ).fetchone()[0]
            )

    def events(self) -> list[tuple[int, str, dict[str, Any]]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT seq, kind, payload FROM journal ORDER BY seq"
            ).fetchall()
        return [
            (int(seq), str(kind), json.loads(payload)) for seq, kind, payload in rows
        ]
