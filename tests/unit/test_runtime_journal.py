import dataclasses
import tempfile
import unittest
import sqlite3
from unittest import mock
from pathlib import Path

from horizon_harness.workspace.journal import ExecutionJournal
from horizon_harness.workspace.store import UnsafeCheckpointError, WorkspaceStore
from horizon_harness.llm.env import FAKE_PROVIDER
from horizon_harness.workspace import TaskSession
from horizon_harness.messages import (
    assistant_message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    tool_results_message,
)
from horizon_harness.workspace.codec import encode_message


class RuntimeJournalTests(unittest.TestCase):
    def test_tool_batch_table_contains_only_immutable_plan_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            with sqlite3.connect(journal.path) as db:
                columns = tuple(
                    row[1] for row in db.execute("PRAGMA table_info(tool_batches)")
                )
            self.assertEqual(columns, ("batch_id", "turn_no", "calls_json"))

    def test_historical_sealed_batch_does_not_authorize_uncommitted_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(Path(tmp) / "store")
            record = store.create("task", task_id="task")
            journal = store.journal(record.task_id)
            assistant = assistant_message((ToolCallBlock("A", "read", {}),))
            journal.plan_batch(
                batch_id="old",
                turn_no=1,
                calls=[{"id": "A", "name": "read", "arguments": {}}],
                assistant_message=encode_message(assistant),
            )
            journal.record_batch_result("old", 0, {"call_id": "A", "tool_result": {}})
            bundle_event = {
                "message": encode_message(
                    tool_results_message(
                        [ToolResultBlock("A", "read", (TextBlock("ok"),))],
                        target="agent",
                    )
                )
            }
            journal.commit_batch_bundle("old", bundle_event)
            journal.set_cursor(turn_no=2, phase="ready")
            store.begin_turn(record.task_id, "new-run", 2)

            with self.assertRaisesRegex(UnsafeCheckpointError, "uncommitted turn"):
                store.assert_resumable(record.task_id)

    def test_reused_provider_id_has_distinct_execution_steps_across_turns(self):
        from horizon_harness.core import Agent, run
        from horizon_harness.state import State
        from horizon_harness.workspace.execution import make_durable_edit_tool

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            (root / "value").write_text("one")
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            answers = iter(
                [
                    assistant_message(
                        (
                            ToolCallBlock(
                                "same-provider-id",
                                "edit",
                                {
                                    "path": "value",
                                    "old_string": old,
                                    "new_string": new,
                                },
                            ),
                        ),
                        sender="agent",
                        kind="step",
                    )
                    for old, new in (("one", "two"), ("two", "three"))
                ]
            )
            agent = Agent(
                name="agent",
                generate=lambda _: next(answers),
                journal=journal,
                tools=(make_durable_edit_tool(root=root, journal=journal),),
            )
            list(run(agent, State(task="edit twice"), max_turns=2))
            self.assertEqual((root / "value").read_text(), "three")
            steps = journal.rebuild_steps()
            self.assertEqual(len(steps), 2)
            self.assertTrue(
                all(step["state"] == "succeeded" for step in steps.values())
            )
            self.assertFalse(any("same-provider-id" in key for key in steps))

    def test_failed_durable_observation_commit_prevents_next_tool_effect(self):
        from horizon_harness.core import dispatch_tool_calls
        from horizon_harness.state import State
        from horizon_harness.tools import AgentTool, text_result

        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.set_cursor(turn_no=1, phase="tools_pending")
            seen = []

            def execute(call_id, arguments, abort, update):
                seen.append(call_id)
                return text_result(call_id)

            tool = AgentTool(
                name="read", description="", parameters={}, execute=execute
            )
            assistant = assistant_message(
                (ToolCallBlock("A", "read", {}), ToolCallBlock("B", "read", {}))
            )
            append = journal.append

            def fail_result(kind, payload):
                if kind == "ToolExecutionEndEvent":
                    raise OSError("observation commit failed")
                return append(kind, payload)

            with mock.patch.object(journal, "append", side_effect=fail_result):
                with self.assertRaisesRegex(OSError, "observation commit failed"):
                    list(
                        dispatch_tool_calls(
                            assistant,
                            {"read": tool},
                            State(task="serial"),
                            journal=journal,
                        )
                    )
            self.assertEqual(seen, ["A"])

    def test_batch_result_failure_does_not_advance_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_batch(batch_id="b", turn_no=1, calls=[{"id": "a"}])
            with self.assertRaises(TypeError):
                journal.record_batch_result("b", 0, {"bad": object()})
            self.assertEqual(journal.batch("b")["next_tool_index"], 0)
            self.assertEqual(journal.batch_results("b"), ())
            journal.record_batch_result("b", 0, {"call_id": "a"})
            self.assertEqual(journal.batch("b")["next_tool_index"], 1)
            with self.assertRaises(ValueError):
                journal.record_batch_result("b", 0, {"call_id": "a"})
            self.assertEqual(len(journal.batch_results("b")), 1)

    def test_model_decision_commit_atomically_publishes_message_batch_steps_and_cursor(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            original = {
                "role": "assistant",
                "sender": "test",
                "target": "user",
                "kind": "step",
                "content": [
                    {
                        "kind": "tool_call",
                        "id": "call-a",
                        "name": "edit",
                        "arguments": {},
                    }
                ],
                "sidecar": {"extra": {"opaque": "preserved"}},
            }
            journal.commit_model_decision(
                events=[
                    ("ModelResponseCommitted", {"index": 1}),
                    ("MessageEvent", {"message": original}),
                ],
                batch_id="batch-1",
                turn_no=2,
                calls=[
                    {
                        "id": "call-a",
                        "execution_step_id": "call-a:edit",
                        "name": "edit",
                        "arguments": {},
                    }
                ],
                assistant_message=original,
            )
            batch = journal.batch("batch-1")
            self.assertEqual(batch["assistant_message"]["role"], "assistant")
            self.assertEqual(journal.step("call-a:edit")["state"], "planned")
            self.assertEqual(journal.cursor()["phase"], "tools_pending")
            message = journal.batch_assistant_message("batch-1")
            self.assertEqual(message.role, "assistant")
            self.assertEqual(message.content[0].id, "call-a")
            self.assertEqual(message.sidecar, original["sidecar"])
            self.assertEqual(batch["assistant_message"]["message_version"], 1)
            self.assertEqual(
                journal.rebuild_steps()["call-a:edit"], journal.step("call-a:edit")
            )
            plans = [
                payload
                for _, kind, payload in journal.events()
                if kind == "ToolBatchPlan"
            ]
            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0]["assistant_message"], batch["assistant_message"])

    def test_model_decision_rejects_missing_message_event_without_writes(self):
        from horizon_harness.messages import assistant_message
        from horizon_harness.workspace.codec import encode_message

        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            with self.assertRaisesRegex(ValueError, "exactly one MessageEvent"):
                journal.commit_model_decision(
                    events=[],
                    batch_id=None,
                    turn_no=1,
                    calls=[],
                    assistant_message=encode_message(assistant_message("done")),
                )
            self.assertEqual(journal.events(), [])
            self.assertEqual(journal.cursor()["phase"], "ready")

    def test_legacy_batch_without_assistant_message_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_batch(
                batch_id="legacy",
                turn_no=1,
                calls=[{"id": "a", "name": "edit", "arguments": {}}],
            )
            with self.assertRaisesRegex(
                ValueError, "MISSING_DURABLE_ASSISTANT_MESSAGE"
            ):
                journal.batch_assistant_message("legacy")

    def test_rebuild_messages_round_trips_canonical_message_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            value = {
                "role": "assistant",
                "sender": "agent",
                "target": "user",
                "kind": "step",
                "content": [{"kind": "text", "text": "ok"}],
            }
            journal.append("MessageEvent", {"message": value})
            self.assertEqual(journal.rebuild_messages()[0].content[0].text, "ok")

    def test_tool_batch_persists_calls_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_batch(
                batch_id="b",
                turn_no=1,
                calls=[{"id": "c", "name": "edit", "arguments": {}}],
            )
            journal.record_batch_result("b", 0, {"call_id": "c", "tool_result": {}})
            bundle_event = {
                "message": encode_message(
                    tool_results_message(
                        [ToolResultBlock("c", "edit", (TextBlock("ok"),))],
                        target="agent",
                    )
                )
            }
            self.assertTrue(journal.commit_batch_bundle("b", bundle_event))
            self.assertTrue(journal.batch("b")["bundle_committed"])
            self.assertIsNone(journal.recoverable_batch("b"))
            self.assertFalse(journal.commit_batch_bundle("b", bundle_event))
            self.assertTrue(journal.batch("b")["bundle_committed"])

    def test_recoverable_batches_lists_unsealed_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_batch(batch_id="b", turn_no=1, calls=[{"id": "a"}])
            self.assertEqual(journal.recoverable_batches()[0]["batch_id"], "b")

    def test_recoverable_batch_returns_only_uncommitted_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            calls = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
            journal.plan_batch(
                batch_id="b",
                turn_no=1,
                calls=calls,
                assistant_message={"role": "assistant", "content": []},
            )
            self.assertEqual(
                journal.batch("b")["assistant_message"]["role"], "assistant"
            )
            journal.record_batch_result("b", 0, {"call_id": "a"})
            self.assertEqual(
                journal.recoverable_batch("b")["remaining_calls"],
                ({"id": "b"}, {"id": "c"}),
            )

    def test_batch_results_must_commit_in_original_call_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_batch(
                batch_id="b",
                turn_no=1,
                calls=[
                    {"id": "a", "name": "one", "arguments": {}},
                    {"id": "b", "name": "two", "arguments": {}},
                ],
            )
            with self.assertRaisesRegex(ValueError, "out of order"):
                journal.record_batch_result("b", 1, {"call_id": "b", "is_error": False})
            journal.record_batch_result("b", 0, {"call_id": "a", "is_error": False})
            journal.record_batch_result("b", 1, {"call_id": "b", "is_error": False})
            self.assertEqual(
                [row["call_id"] for row in journal.rebuild_batch_bundle("b")],
                ["a", "b"],
            )

    def test_unsealed_complete_batch_is_recoverable_without_tool_execution(self):
        from horizon_harness.tools import text_result
        from horizon_harness.workspace.codec import encode_tool_result
        from horizon_harness.workspace.replay import replay_tool_result_blocks

        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_batch(
                batch_id="b",
                turn_no=1,
                calls=[
                    {"id": "a", "name": "read", "arguments": {}},
                    {"id": "b", "name": "read", "arguments": {}},
                ],
                assistant_message={
                    "role": "assistant",
                    "sender": "agent",
                    "target": "user",
                    "kind": "step",
                    "content": [
                        {
                            "kind": "tool_call",
                            "id": "a",
                            "name": "read",
                            "arguments": {},
                        },
                        {
                            "kind": "tool_call",
                            "id": "b",
                            "name": "read",
                            "arguments": {},
                        },
                    ],
                },
            )
            for index, call_id in enumerate(("a", "b")):
                journal.record_batch_result(
                    "b",
                    index,
                    {
                        "call_id": call_id,
                        "tool_result": encode_tool_result(text_result(call_id)),
                    },
                )
            blocks = replay_tool_result_blocks(journal, "b")
            self.assertEqual([block.tool_call_id for block in blocks], ["a", "b"])
            self.assertEqual(journal.recoverable_batch("b")["remaining_calls"], ())
            self.assertEqual(journal.recoverable_batch("b")["remaining_calls"], ())

    def test_three_call_history_rebuild_has_one_assistant_and_ordered_bundle(self):
        from horizon_harness.messages import (
            ToolCallBlock,
            ToolResultBlock,
            TextBlock,
            assistant_message,
            tool_results_message,
        )

        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            assistant = assistant_message(
                (
                    ToolCallBlock("A", "read", {}),
                    ToolCallBlock("B", "read", {}),
                    ToolCallBlock("C", "read", {}),
                ),
                sender="agent",
            )
            journal.append("MessageEvent", {"message": dataclasses.asdict(assistant)})
            bundle = tool_results_message(
                tuple(
                    ToolResultBlock(call_id, "read", (TextBlock(call_id),))
                    for call_id in ("A", "B", "C")
                ),
                target="agent",
            )
            journal.append("MessageEvent", {"message": dataclasses.asdict(bundle)})
            rebuilt = journal.rebuild_messages()
            self.assertEqual(len(rebuilt), 2)
            self.assertEqual(rebuilt[0].tool_calls[0].id, "A")
            self.assertEqual(
                [block.tool_call_id for block in rebuilt[1].content], ["A", "B", "C"]
            )

    def test_cursor_is_persisted_and_reopened(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.set_cursor(turn_no=2, phase="tools_pending", next_tool_index=1)
            reopened = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            self.assertEqual(
                reopened.cursor(),
                {"turn_no": 2, "phase": "tools_pending", "next_tool_index": 1},
            )

    def test_failed_model_batch_commit_does_not_publish_response(self):
        from horizon_harness.core import Agent, run
        from horizon_harness.state import State
        from horizon_harness.messages import assistant_message

        state = State(task="failure")
        sink = mock.Mock()
        sink.events.return_value = []
        sink.commit_model_decision.side_effect = OSError("response commit failure")
        agent = Agent(
            name="test",
            generate=lambda _: assistant_message("answer", sender="test"),
            journal=sink,
        )
        with self.assertRaisesRegex(OSError, "response commit failure"):
            list(run(agent, state, max_turns=1))
        self.assertEqual(state.messages, [])
        self.assertNotIn(
            "ModelResponseEvent", [type(event).__name__ for event in state.events]
        )

    def test_core_commits_exact_public_event_identity(self):
        from horizon_harness.core import Agent, run
        from horizon_harness.state import State

        state = State(task="identity")
        sink = mock.Mock()
        sink.events.return_value = []
        agent = Agent(name="test", generate=mock.Mock(), journal=sink)
        events = run(agent, state, max_turns=1)
        event = next(events)
        payload = sink.append.call_args.args[1]
        self.assertTrue(payload["uuid"])
        self.assertEqual(payload["uuid"], event.uuid)
        self.assertEqual(payload["index"], event.index)
        self.assertEqual(payload["elapsed"], event.elapsed)
        self.assertIs(state.events[-1], event)
        events.close()

    def test_core_commit_failure_does_not_advance_state_or_call_model(self):
        from horizon_harness.core import Agent, run
        from horizon_harness.state import State

        state = State(task="failure")
        sink = mock.Mock()
        sink.events.return_value = []
        sink.append.side_effect = OSError("disk failure")
        generate = mock.Mock()
        agent = Agent(name="test", generate=generate, journal=sink)
        with self.assertRaisesRegex(OSError, "disk failure"):
            next(run(agent, state, max_turns=1))
        self.assertEqual(state.events, [])
        self.assertEqual(state.messages, [])
        generate.assert_not_called()

    def test_task_projection_rebuild_does_not_read_cached_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.commit_projection("WorkspaceUpdated", {}, {"revision": 0})
            journal.append("ModelRequestEvent", {"agent": "test"})
            journal.commit_projection("BoundaryCommitted", {}, {"revision": 1})
            expected = journal.projection()
            with sqlite3.connect(journal.path) as db:
                db.execute("DELETE FROM task_projection")
            self.assertIsNone(journal.projection())
            self.assertEqual(journal.rebuild_task_projection(), expected)

    def test_projection_update_failure_rolls_back_fact_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.commit_projection("WorkspaceUpdated", {}, {"revision": 0})
            before = journal.events()
            with sqlite3.connect(journal.path) as db:
                db.execute("""CREATE TRIGGER reject_projection BEFORE UPDATE ON task_projection
                    BEGIN SELECT RAISE(ABORT, 'injected projection failure'); END""")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "injected"):
                journal.commit_projection("WorkspaceUpdated", {}, {"revision": 1})
            self.assertEqual(journal.events(), before)
            self.assertEqual(journal.projection(), {"revision": 0})
            self.assertEqual(journal.applied_seq(), before[-1][0])

    def test_rebuild_rejects_old_events_without_task_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.append("BoundaryCommitted", {"revision": 1})
            with self.assertRaisesRegex(ValueError, "lacks reconstructable"):
                journal.rebuild_task_projection()

    def test_connections_close_after_success_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            opened = []
            connect = sqlite3.connect

            def tracked_connect(*args, **kwargs):
                db = connect(*args, **kwargs)
                opened.append(db)
                return db

            with mock.patch(
                "horizon_harness.workspace.journal.sqlite3.connect",
                side_effect=tracked_connect,
            ):
                journal.append("ok", {})
                with self.assertRaises(TypeError):
                    journal.append_many([("first", {}), ("bad", {"x": object()})])
                self.assertEqual(len(journal.events()), 1)
            self.assertEqual(len(opened), 3)
            for db in opened:
                with self.assertRaises(sqlite3.ProgrammingError):
                    db.execute("SELECT 1")

    def test_append_is_ordered_and_reopenable(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            self.assertEqual(journal.append("ModelStepPlanned", {"turn": 1}), 1)
            self.assertEqual(
                journal.append_many([("BoundaryCommitted", {"turn": 1})]), 2
            )
            reopened = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            self.assertEqual(reopened.applied_seq(), 2)
            self.assertEqual(
                [kind for _, kind, _ in reopened.events()],
                ["ModelStepPlanned", "BoundaryCommitted"],
            )

    def test_batch_failure_rolls_back_all_events_and_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            with self.assertRaises(TypeError):
                journal.append_many(
                    [
                        ("first", {"ok": 1}),
                        ("bad", {"value": object()}),
                    ]
                )
            self.assertEqual(journal.events(), [])
            self.assertEqual(journal.applied_seq(), 0)

    def test_store_exposes_one_journal_per_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(tmp)
            record = store.create("read only", task_id="task1")
            self.assertEqual(
                store.journal(record.task_id).path, store.journal(record.task_id).path
            )

    def test_task_session_records_core_events_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = TaskSession(
                FAKE_PROVIDER, code_dir=tmp, store_dir=Path(tmp).parent / "store"
            ).start("Return a short answer.", pause_after_turns=1)
            events = (
                WorkspaceStore(Path(tmp).parent / "store")
                .journal(result.task_id)
                .events()
            )
            kinds = [kind for _, kind, _ in events]
            self.assertIn("ModelRequestEvent", kinds)
            self.assertIn("ModelResponseEvent", kinds)
            self.assertIn("BoundaryCommitted", kinds)
