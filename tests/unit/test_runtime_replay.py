import tempfile
import unittest
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

from horizon_harness.workspace.journal import ExecutionJournal
from horizon_harness.workspace.replay import (
    replay,
    fork_workspace,
    fork_from_checkpoint,
    replay_tool_result_blocks,
    diff_runs,
)
from horizon_harness.workspace.snapshot import capture_tree, write_snapshot
from horizon_harness.workspace.models import Checkpoint
from horizon_harness.workspace import TaskSession
from horizon_harness.llm.env import FAKE_PROVIDER
from horizon_harness.llm.types import LLMResponse
from horizon_harness.messages import TextBlock


class RuntimeReplayTests(unittest.TestCase):
    def test_cli_fork_and_diff_use_resumable_task_records(self):
        from runs.task_workspace import main

        with tempfile.TemporaryDirectory() as tmp:
            parent_code = Path(tmp) / "parent-code"
            parent_code.mkdir()
            (parent_code / "value.txt").write_text("x", encoding="utf-8")
            store = Path(tmp) / "store"
            TaskSession(FAKE_PROVIDER, code_dir=parent_code, store_dir=store).start(
                "task", task_id="parent", pause_after_turns=1
            )
            child_code = Path(tmp) / "child-code"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "--code-dir",
                            str(parent_code),
                            "--store-dir",
                            str(store),
                            "--fake",
                            "fork",
                            "parent",
                            "--target-code-dir",
                            str(child_code),
                        ]
                    ),
                    0,
                )
            self.assertEqual(json.loads(output.getvalue())["task_id"], "parent-fork")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "--code-dir",
                            str(parent_code),
                            "--store-dir",
                            str(store),
                            "--fake",
                            "diff",
                            "parent",
                            "parent-fork",
                        ]
                    ),
                    0,
                )
            compared = json.loads(output.getvalue())
            self.assertIn("event_count_delta", compared)

    def test_run_diff_compares_immutable_replay_projections(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = ExecutionJournal(Path(tmp) / "left.sqlite3")
            right = ExecutionJournal(Path(tmp) / "right.sqlite3")
            left.commit_projection(
                "BoundaryCommitted",
                {"turn": 1},
                {
                    "status": "paused",
                    "checkpoint": {"workspace_tree_sha256": "tree-a"},
                },
            )
            right.commit_projection(
                "BoundaryCommitted",
                {"turn": 1},
                {
                    "status": "completed",
                    "checkpoint": {"workspace_tree_sha256": "tree-b"},
                },
            )
            right.append("VerifierProcessObserved", {"outcome": "passed"})

            result = diff_runs(left, right)

            self.assertEqual(
                (result.left_status, result.right_status), ("paused", "completed")
            )
            self.assertEqual(
                (result.left_tree, result.right_tree), ("tree-a", "tree-b")
            )
            self.assertEqual(result.event_count_delta["VerifierProcessObserved"], 1)
            self.assertTrue(result.message_history_equal)

    def test_fork_creates_independent_resumable_child_with_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent_code = Path(tmp) / "parent-code"
            parent_code.mkdir()
            (parent_code / "value.txt").write_text("parent", encoding="utf-8")
            parent_store = Path(tmp) / "parent-store"
            parent_session = TaskSession(
                FAKE_PROVIDER, code_dir=parent_code, store_dir=parent_store
            )
            parent = parent_session.start(
                "continue task", task_id="parent", pause_after_turns=1
            )
            parent_seq = parent_session.store.journal("parent").applied_seq()
            child_code = Path(tmp) / "child-code"
            child_store = Path(tmp) / "child-store"

            child = parent_session.fork_task(
                "parent",
                child_task_id="child",
                child_code_dir=child_code,
                child_store_dir=child_store,
            )

            self.assertEqual(child.parent_task_id, "parent")
            self.assertEqual(child.parent_revision, parent.record.revision)
            self.assertEqual(child.status, "paused")
            self.assertTrue(child.checkpoint.resumable)
            self.assertTrue(
                (
                    child_store / "child" / child.checkpoint.context_manifest_path
                ).is_file()
            )
            self.assertEqual(
                parent_session.store.journal("parent").applied_seq(), parent_seq
            )
            self.assertEqual((parent_code / "value.txt").read_text(), "parent")
            (child_code / "value.txt").write_text("child", encoding="utf-8")
            self.assertEqual((parent_code / "value.txt").read_text(), "parent")
            requests = []

            def model(request):
                requests.append(request)
                return LLMResponse(
                    content=(TextBlock("continued"),),
                    stop_reason="end_turn",
                    model="fake-model",
                )

            TaskSession(
                FAKE_PROVIDER,
                code_dir=child_code,
                store_dir=child_store,
                complete_fn=model,
            ).resume("child", pause_after_turns=1)
            self.assertEqual(len(requests), 1)

    def test_fork_copies_boundary_without_mutating_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            source.mkdir()
            (source / "a.txt").write_text("x", encoding="utf-8")
            parent_hash, child_hash = fork_workspace(source, target)
            self.assertEqual(parent_hash, child_hash)
            self.assertEqual((source / "a.txt").read_text(encoding="utf-8"), "x")
            (target / "a.txt").write_text("y", encoding="utf-8")
            self.assertEqual((source / "a.txt").read_text(encoding="utf-8"), "x")

    def test_replay_reads_events_and_projection_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.commit_projection(
                "BoundaryCommitted", {"turn": 1}, {"status": "paused"}
            )
            before = journal.applied_seq()
            result = replay(journal)
            self.assertEqual(result.projection, {"status": "paused"})
            self.assertEqual(result.events[0][1], "BoundaryCommitted")
            self.assertEqual(journal.applied_seq(), before)

    def test_replay_uses_journal_projection_when_cache_is_tampered(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.commit_projection("WorkspaceUpdated", {}, {"revision": 1})
            with journal._connect() as db:
                db.execute(
                    "UPDATE task_projection SET payload=? WHERE name='task'",
                    ('{"revision": 999}',),
                )
            self.assertEqual(replay(journal).projection, {"revision": 1})

    def test_replay_reconstructs_message_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.append(
                "MessageEvent", {"message": {"role": "user", "content": "one"}}
            )
            journal.append("ModelRequestEvent", {"agent": "test"})
            journal.append(
                "MessageEvent", {"message": {"role": "assistant", "content": "two"}}
            )
            self.assertEqual(
                replay(journal).messages,
                (
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                ),
            )

    def test_checkpoint_carries_workspace_tree_identity(self):
        checkpoint = Checkpoint(workspace_tree_sha256="abc")
        self.assertEqual(checkpoint.workspace_tree_sha256, "abc")

    def test_fork_from_checkpoint_materializes_only_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "task"
            store.mkdir()
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "a.txt").write_text("x")
            snapshot = capture_tree(source, object_dir=store / "snapshot-objects")
            write_snapshot(store / "snap.json", snapshot)
            target = Path(tmp) / "target"
            child_journal = ExecutionJournal(Path(tmp) / "child.sqlite3")
            self.assertEqual(
                fork_from_checkpoint(
                    store,
                    "snap.json",
                    target,
                    child_journal=child_journal,
                    parent_task_id="parent",
                ),
                (snapshot.tree_sha256, snapshot.tree_sha256),
            )
            self.assertEqual((target / "a.txt").read_text(), "x")
            self.assertEqual(child_journal.events()[0][1], "ForkCreated")

    def test_replay_tool_results_are_typed_blocks(self):
        from horizon_harness.tools import text_result
        from horizon_harness.workspace.codec import encode_tool_result

        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_batch(
                batch_id="b",
                turn_no=1,
                calls=[{"id": "a", "name": "read", "arguments": {}}],
            )
            original = text_result("actual file contents", is_error=True)
            journal.record_batch_result(
                "b", 0, {"call_id": "a", "tool_result": encode_tool_result(original)}
            )
            block = replay_tool_result_blocks(journal, "b")[0]
            self.assertEqual(block.tool_name, "read")
            self.assertEqual(block.content, original.content)
            self.assertTrue(block.is_error)

    def test_legacy_result_summary_cannot_become_tool_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_batch(
                batch_id="b",
                turn_no=1,
                calls=[{"id": "a", "name": "read", "arguments": {}}],
            )
            journal.record_batch_result("b", 0, {"call_id": "a", "is_error": False})
            with self.assertRaisesRegex(ValueError, "MISSING_DURABLE_TOOL_RESULT"):
                replay_tool_result_blocks(journal, "b")
