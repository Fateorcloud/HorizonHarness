import tempfile
import unittest
import os
import subprocess
import sys
import sqlite3
from pathlib import Path
from unittest import mock

from horizon_harness.workspace.journal import ExecutionJournal
from horizon_harness.workspace.execution import (
    reconcile_edit,
    atomic_edit_write,
)
from horizon_harness.tools.edit import edit_fingerprint


class ToolRecoveryJournalTests(unittest.TestCase):
    def test_preplanned_edit_persists_intent_before_writing(self):
        from horizon_harness.tools import tool_arguments_hash
        from horizon_harness.workspace.execution import make_durable_edit_tool

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            (root / "a").write_text("before")
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            args = {"path": "a", "old_string": "before", "new_string": "after"}
            journal.plan_step(
                step_id="call:edit",
                tool_name="edit",
                input_hash=tool_arguments_hash(args),
                idempotency_key="k",
                effect_class="idempotent_effect",
            )
            tool = make_durable_edit_tool(root=root, journal=journal)

            def check_then_write(path, content):
                intent = ExecutionJournal(journal.path).step("call:edit")[
                    "result_payload"
                ]
                self.assertEqual(intent["before"], edit_fingerprint("before"))
                self.assertEqual(intent["after"], edit_fingerprint("after"))
                atomic_edit_write(path, content)

            with mock.patch(
                "horizon_harness.workspace.execution.atomic_edit_write",
                side_effect=check_then_write,
            ):
                original = tool.execute("call", args, lambda: False, None)
                self.assertFalse(original.is_error)
            with mock.patch(
                "horizon_harness.workspace.execution.atomic_edit_write"
            ) as write:
                self.assertEqual(
                    tool.execute("call", args, lambda: False, None), original
                )
                write.assert_not_called()
            self.assertEqual((root / "a").read_text(), "after")

    def test_argument_hash_is_stable_across_processes_and_nested_key_order(self):
        from horizon_harness.tools import tool_arguments_hash

        expected = tool_arguments_hash({"z": {"b": 2, "a": 1}, "path": "文件"})
        self.assertEqual(
            expected, tool_arguments_hash({"path": "文件", "z": {"a": 1, "b": 2}})
        )
        script = "from horizon_harness.tools import tool_arguments_hash; print(tool_arguments_hash({'path':'文件','z':{'a':1,'b':2}}))"
        for seed in ("1", "2"):
            process = subprocess.run(
                [sys.executable, "-c", script],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(process.stdout.strip(), expected)
        with self.assertRaises(ValueError):
            tool_arguments_hash({"bad": float("nan")})

    def test_atomic_edit_keeps_original_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "code.txt"
            path.write_text("before")
            path.chmod(0o640)
            with mock.patch(
                "horizon_harness.workspace.execution.os.replace",
                side_effect=OSError("injected"),
            ):
                with self.assertRaises(OSError):
                    atomic_edit_write(path, "after")
            self.assertEqual(path.read_text(), "before")
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            atomic_edit_write(path, "after")
            self.assertEqual(path.read_text(), "after")
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_retry_authorization_persists_budget_and_rejects_stale_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.sqlite3"
            journal = ExecutionJournal(path)
            journal.plan_step(
                step_id="a",
                tool_name="edit",
                input_hash="h",
                idempotency_key="k",
                effect_class="idempotent_effect",
            )
            self.assertEqual(
                journal.begin_retry("a", expected_attempt=1, max_attempts=2), 2
            )
            reopened = ExecutionJournal(path)
            before = reopened.events()
            with self.assertRaisesRegex(ValueError, "retry requires"):
                reopened.begin_retry("a", expected_attempt=1, max_attempts=2)
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                reopened.begin_retry("a", expected_attempt=2, max_attempts=2)
            self.assertEqual(reopened.events(), before)
            self.assertEqual(reopened.rebuild_steps()["a"]["attempt"], 2)

    def test_reconciliation_rejects_path_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.write_text("before")
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_step(
                step_id="a",
                tool_name="edit",
                input_hash="h",
                idempotency_key="k",
                effect_class="idempotent_effect",
                result_payload={
                    "path": "../outside",
                    "before": edit_fingerprint("before"),
                    "after": edit_fingerprint("after"),
                    "old_string": "before",
                    "new_string": "after",
                    "replace_all": False,
                },
            )
            self.assertEqual(
                reconcile_edit(journal, root=root, retry_before=True), ("ambiguous:a",)
            )
            self.assertEqual(outside.read_text(), "before")
            self.assertEqual(journal.step("a")["attempt"], 1)

    def test_steps_rebuild_without_projection_and_failed_updates_roll_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_step(
                step_id="a",
                tool_name="edit",
                input_hash="h",
                idempotency_key="k",
                effect_class="idempotent_effect",
            )
            journal.commit_step_result(
                "a", state="succeeded", result_payload={"after": "x"}
            )
            expected = journal.step("a")
            before = journal.events()
            with self.assertRaises(KeyError):
                journal.commit_step_result("missing", state="failed", result_payload={})
            self.assertEqual(journal.events(), before)
            self.assertEqual(journal.applied_seq(), before[-1][0])
            with sqlite3.connect(journal.path) as db:
                db.execute("DELETE FROM execution_steps")
            self.assertEqual(journal.rebuild_steps(), {"a": expected})

    def test_step_identity_is_persisted_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_step(
                step_id="task:step:1",
                tool_name="edit",
                input_hash="input-hash",
                idempotency_key="task:step:1:edit:v1",
                effect_class="idempotent_effect",
            )
            step = journal.step("task:step:1")
            self.assertIsNotNone(step)
            self.assertEqual(step["state"], "planned")
            self.assertEqual(step["attempt"], 1)

    def test_idempotency_key_is_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            args = dict(
                tool_name="edit",
                input_hash="same",
                idempotency_key="same-key",
                effect_class="idempotent_effect",
            )
            journal.plan_step(step_id="a", **args)
            with self.assertRaises(Exception):
                journal.plan_step(step_id="b", **args)

    def test_step_result_is_terminal_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_step(
                step_id="a",
                tool_name="edit",
                input_hash="h",
                idempotency_key="k",
                effect_class="idempotent_effect",
            )
            journal.commit_step_result(
                "a", state="succeeded", result_payload={"after": "x"}
            )
            self.assertEqual(journal.step("a")["state"], "succeeded")

    def test_reconcile_after_tree_does_not_repeat_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            target = root / "a.txt"
            target.write_text("after", encoding="utf-8")
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_step(
                step_id="a",
                tool_name="edit",
                input_hash="h",
                idempotency_key="k",
                effect_class="idempotent_effect",
            )
            journal.commit_step_result(
                "a",
                state="planned",
                result_payload={
                    "path": "a.txt",
                    "before": edit_fingerprint("before"),
                    "after": edit_fingerprint("after"),
                },
            )
            self.assertEqual(reconcile_edit(journal, root=root), ("succeeded:a",))
            self.assertEqual(journal.step("a")["state"], "succeeded")

    def test_reconcile_before_tree_retries_once_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            (root / "a.txt").write_text("before", encoding="utf-8")
            journal = ExecutionJournal(Path(tmp) / "journal.sqlite3")
            journal.plan_step(
                step_id="a",
                tool_name="edit",
                input_hash="h",
                idempotency_key="k",
                effect_class="idempotent_effect",
            )
            journal.commit_step_result(
                "a",
                state="planned",
                result_payload={
                    "path": "a.txt",
                    "before": edit_fingerprint("before"),
                    "after": edit_fingerprint("after"),
                    "old_string": "before",
                    "new_string": "after",
                    "replace_all": False,
                },
            )
            self.assertEqual(
                reconcile_edit(journal, root=root, retry_before=True), ("succeeded:a",)
            )
            self.assertEqual((root / "a.txt").read_text(encoding="utf-8"), "after")

    def test_subprocess_crash_after_edit_reconciles_without_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            target = root / "a.txt"
            target.write_text("before", encoding="utf-8")
            db = Path(tmp) / "journal.sqlite3"
            script = """
from horizon_harness.workspace.execution import make_durable_edit_tool
from horizon_harness.workspace.journal import ExecutionJournal
from pathlib import Path
import os
root=Path(os.environ['HH_ROOT']); j=ExecutionJournal(os.environ['HH_DB'])
tool=make_durable_edit_tool(root=root, journal=j)
tool.execute('crash-call', {'path':'a.txt','old_string':'before','new_string':'after'}, lambda:False, None)
"""
            env = {
                **os.environ,
                "HH_ROOT": str(root),
                "HH_DB": str(db),
                "HORIZON_CRASH_AFTER_EDIT": "1",
            }
            proc = subprocess.run([sys.executable, "-c", script], env=env)
            self.assertEqual(proc.returncode, 91)
            journal = ExecutionJournal(db)
            self.assertEqual(
                reconcile_edit(journal, root=root), ("succeeded:crash-call:edit",)
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "after")
