from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from horizon_harness.context_view import ContextPolicy, build_context_view
from horizon_harness.core import Agent
from horizon_harness.state import State
from horizon_harness.llm.env import FAKE_PROVIDER
from horizon_harness.llm.types import LLMResponse, TokenUsage
from horizon_harness.memory import (
    ExtractedMemory,
    IndexedMemory,
    Memory,
    MemoryContext,
    MemoryExtractionResult,
    memory_context_message,
)
from horizon_harness.messages import (
    ToolCallBlock,
    assistant_message,
    runtime_message,
    text_of,
    tool_result_message,
)
from horizon_harness.protocols import ModelRequestEvent
from horizon_harness.workspace import (
    Artifact,
    Checkpoint,
    TaskSession,
    TaskStep,
    UnsafeCheckpointError,
    WorkspaceContextBuilder,
    WorkspaceStore,
)
from horizon_harness.workspace.tools import (
    make_workspace_update_tool,
    make_workspace_read_tool,
)

from tests.unit._support import execute_tool


ROOT = Path(__file__).resolve().parents[2]


class WorkspaceStoreTests(unittest.TestCase):
    def test_json_round_trip_and_schema_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(tmp)
            created = store.create(
                "fix it", constraints=("no network",), acceptance=("tests pass",)
            )
            revised = replace(
                created,
                revision=1,
                steps=(TaskStep("s1", "inspect", "done", ("r:0",)),),
            )
            store.save(revised)
            self.assertEqual(store.load(created.task_id), revised)

            path = store.task_dir(created.task_id) / "task.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["schema_version"] = 99
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                store.load(created.task_id)

    def test_atomic_save_failure_keeps_previous_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(tmp)
            created = store.create("fix it")
            path = store.task_dir(created.task_id) / "task.json"
            before = path.read_bytes()
            with mock.patch(
                "horizon_harness.workspace.store.os.replace",
                side_effect=OSError("injected"),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    store.save(replace(created, revision=1))
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_name("task.json.part").exists())

    def test_uncommitted_turn_refuses_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(tmp)
            created = store.create("fix it")
            store.begin_turn(created.task_id, "run", 1)
            with self.assertRaisesRegex(UnsafeCheckpointError, "side effects"):
                store.assert_resumable(created.task_id)


class CombinedContextTests(unittest.TestCase):
    def test_current_state_wins_and_details_are_read_on_demand(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(tmp)
            record = store.create("current goal", constraints=("keep API",))
            record = replace(
                record,
                revision=2,
                next_action="run tests",
                steps=(TaskStep(id="old", description="detail-only", status="done"),),
            )
            store.save(record)
            state = State(task=record.task)
            state.record(runtime_message(record.task, kind="task"))
            builder = WorkspaceContextBuilder(
                store,
                record.task_id,
                representation="combined",
                summary="Old suggestion: change API",
                summary_revision=1,
            )
            view = builder("a", state, ContextPolicy())
            text = "\n".join(text_of(m.content) for m in view.messages)
            self.assertIn("keep API", text)
            self.assertIn("stale=True", text)
            self.assertIn("run tests", text)
            self.assertNotIn("detail-only", text)
            tool = make_workspace_read_tool(store, record.task_id)
            result = tool.execute(
                "read", {"section": "steps"}, lambda: False, lambda _: None
            )
            self.assertIn("detail-only", text_of(result.content))
            with self.assertRaisesRegex(ValueError, "newer"):
                replace(builder, summary_revision=3)("a", state, ContextPolicy())

    def test_combined_resume_in_another_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = TaskSession(FAKE_PROVIDER, code_dir=ROOT, store_dir=tmp).start(
                "Return done", pause_after_turns=1
            )
            code = (
                "from horizon_harness.workspace import TaskSession; "
                "from horizon_harness.llm.env import FAKE_PROVIDER; "
                f"s=TaskSession(FAKE_PROVIDER,code_dir={str(ROOT)!r},store_dir={tmp!r},"
                "context_representation='combined',continuation_summary='Continue task',"
                f"summary_revision={result.record.revision}); "
                f"r=s.resume({result.task_id!r},pause_after_turns=1); print(r.run_id)"
            )
            child = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            )
            self.assertNotEqual(child.stdout.strip(), result.run_id)


class WorkspaceToolTests(unittest.TestCase):
    def test_update_checks_revision_references_and_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp)
            (code / "result.txt").write_text("ok", encoding="utf-8")
            holder = [WorkspaceStore(code / "store").create("fix")]
            tool = make_workspace_update_tool(
                current=lambda: holder[0],
                update=lambda value: holder.__setitem__(0, value),
                code_dir=code,
            )
            conflict = execute_tool(tool, {"expected_revision": 7})
            self.assertTrue(conflict.is_error)
            bad_ref = execute_tool(
                tool,
                {
                    "expected_revision": 0,
                    "artifacts": [{"path": "result.txt", "step_ids": ["missing"]}],
                },
            )
            self.assertTrue(bad_ref.is_error)
            good = execute_tool(
                tool,
                {
                    "expected_revision": 0,
                    "steps": [{"id": "s1", "description": "write", "status": "done"}],
                    "artifacts": [{"path": "result.txt", "step_ids": ["s1"]}],
                    "next_action": "verify",
                },
            )
            self.assertFalse(good.is_error)
            self.assertEqual(holder[0].revision, 1)
            self.assertEqual(
                holder[0].artifacts[0].sha256,
                WorkspaceStore.sha256(code / "result.txt"),
            )

            escaped = execute_tool(
                tool,
                {"expected_revision": 1, "artifacts": [{"path": "../outside"}]},
            )
            self.assertTrue(escaped.is_error)


class WorkspaceSessionTests(unittest.TestCase):
    def test_rebuild_state_uses_journal_canonical_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = TaskSession(FAKE_PROVIDER, code_dir=ROOT, store_dir=tmp)
            result = session.start("Return done.", pause_after_turns=1)
            rebuilt = session.rebuild_state(result.task_id)
            self.assertEqual(len(rebuilt.messages), len(result.state.messages))

    def test_task_session_has_no_second_recovery_execution_entry_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = TaskSession(FAKE_PROVIDER, code_dir=ROOT, store_dir=tmp)
            self.assertFalse(hasattr(session, "recover_batch"))
            self.assertFalse(hasattr(session, "recover_batch_message"))

    def test_start_then_new_process_resume_uses_disk_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = TaskSession(FAKE_PROVIDER, code_dir=ROOT, store_dir=tmp)
            first = session.start("Return done.", pause_after_turns=1)
            self.assertEqual(first.status, "paused")
            self.assertTrue(first.record.checkpoint.resumable)
            code = (
                "import json; "
                "from horizon_harness.llm.env import FAKE_PROVIDER; "
                "from horizon_harness.workspace import TaskSession; "
                f"s=TaskSession(FAKE_PROVIDER,code_dir={str(ROOT)!r},store_dir={tmp!r}); "
                f"r=s.resume({first.task_id!r},pause_after_turns=1); "
                "print(json.dumps({'run_id':r.run_id,'messages':len(r.state.messages)}))"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            resumed = json.loads(completed.stdout)
            self.assertNotEqual(resumed["run_id"], first.run_id)
            self.assertGreater(resumed["messages"], len(first.state.messages))
            old_ref = first.record.checkpoint.message_refs[0]
            self.assertIn(
                "Return done",
                session.store.history_read(first.task_id, [old_ref], 2_000),
            )

    def test_resume_rejects_changed_provider_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = TaskSession(FAKE_PROVIDER, code_dir=ROOT, store_dir=tmp)
            first = session.start("Return done.", pause_after_turns=1)
            changed = replace(FAKE_PROVIDER, model="different")
            with self.assertRaisesRegex(
                UnsafeCheckpointError, "provider configuration"
            ):
                TaskSession(changed, code_dir=ROOT, store_dir=tmp).resume(first.task_id)

    def test_workspace_and_memory_can_be_enabled_together(self) -> None:
        class InitialMemory(Memory):
            def initial(self, ctx: MemoryContext):
                return (memory_context_message("remembered", target=ctx.agent),)

        with tempfile.TemporaryDirectory() as tmp:
            result = TaskSession(
                FAKE_PROVIDER,
                code_dir=ROOT,
                store_dir=tmp,
                memory=InitialMemory(),
                memory_scope=str(ROOT),
            ).start("Return done.", pause_after_turns=1)
            request = next(
                event for event in result.events if isinstance(event, ModelRequestEvent)
            )
            payload_text = " ".join(
                text_of(message.content) for message in request.llm_payload
            )
            self.assertIn("remembered", payload_text)
            self.assertIn("TASK WORKSPACE", payload_text)
            self.assertEqual(
                request.context_view["fixed_overhead_tools"],
                [
                    "read",
                    "edit",
                    "workspace_update",
                    "history_read",
                ],
            )

    def test_context_budget_uses_provider_window_and_reserves_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = replace(FAKE_PROVIDER, context_window=5_000)
            result = TaskSession(
                provider,
                code_dir=ROOT,
                store_dir=tmp,
                context_budget=12_000,
                output_reserve=2_048,
                safety_buffer=512,
            ).start("Return done.", pause_after_turns=1)
            request = next(
                event for event in result.events if isinstance(event, ModelRequestEvent)
            )
            self.assertEqual(request.context_view["max_input_tokens"], 2_440)
            self.assertEqual(request.context_view["model_context_window"], 5_000)
            self.assertEqual(request.context_view["output_reserve"], 2_048)
            self.assertEqual(request.context_view["safety_buffer"], 512)

    def test_summary_resume_uses_same_checkpoint_without_workspace_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = TaskSession(FAKE_PROVIDER, code_dir=ROOT, store_dir=tmp).start(
                "Return done.", pause_after_turns=1
            )
            resumed = TaskSession(
                FAKE_PROVIDER,
                code_dir=ROOT,
                store_dir=tmp,
                context_representation="summary",
                continuation_summary="The first complete turn inspected the repository.",
            ).resume(first.task_id, pause_after_turns=1)
            request = next(
                event
                for event in resumed.events
                if isinstance(event, ModelRequestEvent)
            )

            self.assertEqual(request.context_view["builder"], "summary")
            self.assertIn(
                "CONTINUATION SUMMARY",
                " ".join(text_of(message.content) for message in request.llm_payload),
            )
            self.assertNotIn(
                "workspace_update", [tool["name"] for tool in request.tools]
            )
            self.assertIn("history_read", [tool["name"] for tool in request.tools])

    def test_completed_task_extracts_validated_memory_from_final_evidence(self) -> None:
        class CapturingExtractor:
            evidence = None

            def extract(self, evidence):
                self.evidence = evidence
                return MemoryExtractionResult(
                    (
                        ExtractedMemory(
                            type="fact",
                            content="completed tasks preserve sourced evidence",
                            source_ref=f"workspace:{evidence.source_revision}",
                        ),
                    )
                )

        extractor = CapturingExtractor()
        responses = (
            LLMResponse(
                content=(
                    ToolCallBlock(
                        "finish",
                        "workspace_update",
                        {"expected_revision": 0, "status": "completed"},
                    ),
                ),
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                model="fake-model",
            ),
            LLMResponse(
                content=(runtime_message("done").content[0],),
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=20, output_tokens=2),
                model="fake-model",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            memory = IndexedMemory(Path(tmp) / "memory.sqlite3")
            session = TaskSession(
                FAKE_PROVIDER,
                code_dir=ROOT,
                store_dir=Path(tmp) / "workspace",
                memory=memory,
                memory_scope="repo",
                memory_extractor=extractor,
                complete_fn=mock.Mock(side_effect=responses),
            )
            result = session.start("finish the task")

            self.assertEqual(result.status, "paused")
            self.assertEqual(len(result.memory_entry_ids), 0)
            self.assertEqual(result.memory_extraction_error, "")
            self.assertIsNone(extractor.evidence)
            self.assertEqual(memory.search("sourced evidence", "repo"), ())

    def test_modified_artifact_is_preserved_and_requires_reverification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp) / "code"
            code.mkdir()
            artifact_path = code / "answer.txt"
            artifact_path.write_text("one", encoding="utf-8")
            session = TaskSession(
                FAKE_PROVIDER, code_dir=code, store_dir=Path(tmp) / "store"
            )
            first = session.start("Return done.", pause_after_turns=1)
            record = replace(
                first.record,
                artifacts=(
                    Artifact(
                        "answer.txt",
                        WorkspaceStore.sha256(artifact_path),
                        verification_refs=("verify:1",),
                    ),
                ),
            )
            session.store.save(record)
            artifact_path.write_text("two", encoding="utf-8")
            resumed = session.resume(first.task_id, pause_after_turns=1)
            self.assertEqual(artifact_path.read_text(encoding="utf-8"), "two")
            self.assertEqual(resumed.record.artifacts[0].verification_refs, ())
            self.assertIn("requires verification", resumed.record.blockers[0])


class ContextBuilderTests(unittest.TestCase):
    def test_custom_builder_is_the_payload_recorded_in_trace(self) -> None:
        marker = runtime_message("only this", target="a", kind="context")

        def builder(name, state, policy):
            del state
            return build_context_view(name, [marker], policy=policy)

        agent = Agent(
            "a",
            lambda visible: assistant_message("done", sender="a", kind="final"),
            context_builder=builder,
        )
        state, events = agent.run("hidden original")
        tuple(events)
        request = next(
            event for event in state.events if isinstance(event, ModelRequestEvent)
        )
        self.assertEqual(request.visible_count, 1)
        self.assertEqual(text_of(request.llm_payload[0].content), "only this")

    def test_workspace_context_keeps_recent_tool_pair_and_archives_old_turn(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(tmp)
            record = store.create("goal", constraints=("keep",))
            record = replace(
                record,
                checkpoint=Checkpoint(run_id="run", resumable=True),
            )
            store.save(record)
            state = Agent(
                "a", lambda _: assistant_message("x", sender="a")
            )._default_init_state("goal")
            state.record(assistant_message("old", sender="a", kind="step"))
            state.record(assistant_message("recent1", sender="a", kind="step"))
            state.record(assistant_message("recent2", sender="a", kind="step"))
            view = WorkspaceContextBuilder(store, record.task_id, run_id="run")(
                "a", state, ContextPolicy()
            )
            self.assertIn("goal", view.messages[0].content[0].text)
            self.assertEqual(
                [m.content[0].text for m in view.messages[-2:]], ["recent1", "recent2"]
            )
            self.assertEqual(view.metadata["archive_refs"], ["run:1"])

    def test_budget_trimming_drops_a_complete_turn_and_updates_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkspaceStore(tmp)
            record = store.create("goal")
            store.save(
                replace(
                    record,
                    checkpoint=Checkpoint(run_id="run", resumable=True),
                )
            )
            state = Agent(
                "a", lambda _: assistant_message("x", sender="a")
            )._default_init_state("goal")
            state.record(
                assistant_message(
                    (ToolCallBlock("old", "read", {"path": "x" * 2_000}),),
                    sender="a",
                    kind="step",
                )
            )
            state.record(
                tool_result_message(
                    "old result",
                    tool_call_id="old",
                    tool_name="read",
                    target="a",
                )
            )
            state.record(
                assistant_message(
                    (ToolCallBlock("new", "read", {"path": "small"}),),
                    sender="a",
                    kind="step",
                    usage=TokenUsage(input_tokens=50000, output_tokens=20),
                )
            )
            state.record(
                tool_result_message(
                    "new result",
                    tool_call_id="new",
                    tool_name="read",
                    target="a",
                )
            )

            view = WorkspaceContextBuilder(
                store,
                record.task_id,
                max_input_tokens=400,
                run_id="run",
            )("a", state, ContextPolicy())

            self.assertEqual(view.metadata["archive_refs"], ["run:1", "run:2"])
            self.assertEqual(view.metadata["included_message_refs"], ["run:3", "run:4"])
            self.assertEqual(
                [message.kind for message in view.messages[-2:]],
                ["step", "tool_result"],
            )


if __name__ == "__main__":
    unittest.main()
