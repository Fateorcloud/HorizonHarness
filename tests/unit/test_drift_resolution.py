"""Human drift resolution preserves file bytes and canonical tool history."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from horizon_harness.llm.env import FAKE_PROVIDER
from horizon_harness.llm.types import LLMResponse
from horizon_harness.messages import TextBlock, ToolCallBlock
from horizon_harness.tools import text_result
from horizon_harness.workspace import TaskSession
from horizon_harness.workspace.codec import encode_tool_result
from horizon_harness.workspace.execution import (
    atomic_edit_write,
    make_durable_edit_tool,
)
from horizon_harness.workspace.journal import ExecutionJournal
from horizon_harness.workspace.store import UnsafeCheckpointError
from runs.task_workspace import main


class Interrupted(BaseException):
    pass


def sha(content):
    return hashlib.sha256(content).hexdigest()


def response(*calls):
    return LLMResponse(
        content=tuple(calls) or (TextBlock("done"),),
        stop_reason="tool_use" if calls else "end_turn",
        model="fake-model",
    )


class DriftResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.code = self.base / "code"
        self.code.mkdir()
        self.store = self.base / "store"
        self.before = b"value=old\nkeep\n"
        self.after = b"value=new\nkeep\n"
        self.current = b"value=external\nkeep\n"

    def session(self, model=None):
        return TaskSession(
            FAKE_PROVIDER,
            code_dir=self.code,
            store_dir=self.store,
            complete_fn=model or (lambda request: response()),
        )

    def interrupt_edit(self, *, extra_calls=()):
        for name in ("a", "b", "c"):
            (self.code / name).write_bytes(self.before)
        calls = (
            tuple(
                ToolCallBlock(
                    name.upper(),
                    "edit",
                    {"path": name, "old_string": "old", "new_string": "new"},
                )
                for name in ("a", "b", "c")
            )
            + extra_calls
        )

        def write_then_interrupt(path, content):
            atomic_edit_write(path, content)
            if path.name == "b":
                raise Interrupted()

        with mock.patch(
            "horizon_harness.workspace.execution.atomic_edit_write",
            side_effect=write_then_interrupt,
        ):
            with self.assertRaises(Interrupted):
                self.session(lambda request: response(*calls)).start(
                    "edit files", task_id="drift"
                )
        (self.code / "b").write_bytes(self.current)
        self.journal = self.session().store.journal("drift")
        self.batch = self.journal.batch(self.journal.cursor()["batch_id"])
        self.step_id = self.batch["calls"][1]["execution_step_id"]

    def prepare_drift(self, **kwargs):
        self.interrupt_edit(**kwargs)
        with self.assertRaisesRegex(UnsafeCheckpointError, "workspace drift"):
            self.session().resume("drift")

    def report(self):
        from horizon_harness.workspace.drift import drift_report

        return drift_report(self.journal, root=self.code, step_id=self.step_id)

    def adopt(self, expected_sha=None, note=""):
        return self.session().adopt_current(
            "drift",
            step_id=self.step_id,
            expected_sha=expected_sha or sha(self.current),
            note=note,
        )

    def test_report_has_three_hashes_and_both_diffs(self):
        self.prepare_drift()
        events = self.journal.events()
        report = self.report()
        self.assertEqual(report["path"], "b")
        self.assertEqual(report["before_sha256"], sha(self.before))
        self.assertEqual(report["after_sha256"], sha(self.after))
        self.assertEqual(report["current_sha256"], sha(self.current))
        self.assertTrue(report["before_image_available"])
        self.assertIn("-value=old\n+value=external", report["before_to_current_diff"])
        self.assertIn("-value=new\n+value=external", report["after_to_current_diff"])
        self.assertFalse(report["old_string_present"])
        self.assertFalse(report["new_string_present"])
        self.assertEqual(self.journal.events(), events)

    def test_legacy_report_without_before_object_degrades(self):
        self.prepare_drift()
        (self.journal.path.parent / "snapshot-objects" / sha(self.before)).unlink()
        (self.code / "b").write_bytes(b"old and new\n")
        report = self.report()
        self.assertFalse(report["before_image_available"])
        self.assertEqual(report["current_sha256"], sha(b"old and new\n"))
        self.assertIsNone(report["before_to_current_diff"])
        self.assertIsNone(report["after_to_current_diff"])
        self.assertEqual(report["diff_unavailable_reason"], "before_image_missing")
        self.assertTrue(report["old_string_present"])
        self.assertTrue(report["new_string_present"])

    def test_report_omits_after_diff_when_reconstructed_hash_disagrees(self):
        self.prepare_drift()
        step = self.journal.step(self.step_id)
        self.journal.set_step_state(
            self.step_id,
            "ambiguous",
            {**step["result_payload"], "after": sha(b"wrong")},
        )
        report = self.report()
        self.assertIsNotNone(report["before_to_current_diff"])
        self.assertIsNone(report["after_to_current_diff"])
        self.assertEqual(report["after_diff_unavailable_reason"], "after_hash_mismatch")

    def test_report_skips_binary_and_invalid_utf8_diffs(self):
        self.prepare_drift()
        for content in (b"old\0new", b"old\xffnew"):
            with self.subTest(content=content):
                (self.code / "b").write_bytes(content)
                report = self.report()
                self.assertTrue(report["binary"])
                self.assertIsNone(report["before_to_current_diff"])
                self.assertIsNone(report["after_to_current_diff"])
                self.assertEqual(report["diff_unavailable_reason"], "binary")

    def test_report_rejects_corrupt_before_image(self):
        self.prepare_drift()
        (self.journal.path.parent / "snapshot-objects" / sha(self.before)).write_bytes(
            b"corrupt"
        )
        report = self.report()
        self.assertFalse(report["before_image_available"])
        self.assertEqual(
            report["diff_unavailable_reason"], "before_image_hash_mismatch"
        )
        self.assertIsNone(report["after_to_current_diff"])

    def test_report_reconstructs_replace_all_and_empty_old_string(self):
        self.prepare_drift()
        payload = self.journal.step(self.step_id)["result_payload"]
        cases = (
            (b"old old\n", "old", "new", True, b"new new\n"),
            (b"", "", "created\n", False, b"created\n"),
        )
        for before, old, new, replace_all, after in cases:
            with self.subTest(before=before):
                obj = self.journal.path.parent / "snapshot-objects" / sha(before)
                obj.write_bytes(before)
                self.journal.set_step_state(
                    self.step_id,
                    "ambiguous",
                    {
                        **payload,
                        "before": sha(before),
                        "after": sha(after),
                        "old_string": old,
                        "new_string": new,
                        "replace_all": replace_all,
                    },
                )
                report = self.report()
                self.assertIn(after.decode().rstrip(), report["after_to_current_diff"])
                self.assertNotIn("after_diff_unavailable_reason", report)

    def test_report_preserves_missing_final_newline_in_diff(self):
        self.prepare_drift()
        (self.code / "b").write_bytes(self.before.rstrip(b"\n"))
        self.assertIn(
            "\\ No newline at end of file", self.report()["before_to_current_diff"]
        )

    def test_deleted_file_can_be_adopted_without_recreating_it(self):
        self.prepare_drift()
        (self.code / "b").unlink()
        report = self.report()
        self.assertFalse(report["current_exists"])
        self.assertEqual(report["current_sha256"], sha(b""))
        with mock.patch(
            "horizon_harness.workspace.execution.atomic_edit_write"
        ) as writes:
            self.adopt(expected_sha=sha(b""))
            self.session().resume("drift")
            writes.assert_not_called()
        self.assertFalse((self.code / "b").exists())

    def test_report_and_adopt_refuse_symlink_paths(self):
        self.prepare_drift()
        outside = self.base / "outside"
        outside.write_bytes(self.current)
        (self.code / "b").unlink()
        (self.code / "b").symlink_to(outside)
        before = self.journal.events()
        with self.assertRaisesRegex(ValueError, "unsafe edit path"):
            self.report()
        with self.assertRaisesRegex(ValueError, "unsafe edit path"):
            self.adopt()
        self.assertEqual(self.journal.events(), before)
        self.assertEqual(outside.read_bytes(), self.current)

    def test_before_image_publish_failure_leaves_no_intent_or_temporary_object(self):
        from horizon_harness.workspace.drift import store_before_image

        journal = ExecutionJournal(self.store / "journal.sqlite3")
        with mock.patch(
            "horizon_harness.workspace.drift.os.replace",
            side_effect=OSError("injected"),
        ):
            with self.assertRaisesRegex(OSError, "injected"):
                store_before_image(journal, self.before)
        self.assertEqual(list((journal.path.parent / "snapshot-objects").iterdir()), [])
        self.assertEqual(journal.events(), [])
        store_before_image(journal, self.before)
        with mock.patch("horizon_harness.workspace.drift.os.replace") as replace:
            store_before_image(journal, self.before)
            replace.assert_not_called()

    def test_before_image_is_atomic_and_precedes_edit_intent(self):
        target = self.code / "b"
        target.write_bytes(self.before)
        journal = ExecutionJournal(self.store / "journal.sqlite3")
        original = journal._append_step_event
        obj = journal.path.parent / "snapshot-objects" / sha(self.before)

        def check_image(db, kind, payload):
            if kind in ("StepPlanned", "StepResultCommitted") and (
                payload.get("result_payload") or {}
            ).get("before"):
                self.assertEqual(obj.read_bytes(), self.before)
                self.assertEqual(list(obj.parent.iterdir()), [obj])
                raise Interrupted()
            original(db, kind, payload)

        with mock.patch.object(journal, "_append_step_event", side_effect=check_image):
            with self.assertRaises(Interrupted):
                make_durable_edit_tool(root=self.code, journal=journal).execute(
                    "B",
                    {"path": "b", "old_string": "old", "new_string": "new"},
                    lambda: False,
                    None,
                )
        self.assertEqual(target.read_bytes(), self.before)
        step = journal.step("B:edit")
        self.assertTrue(step is None or step["result_payload"] is None)

    def test_adopt_then_resume_preserves_bytes_writes_nothing_and_pairs_history(self):
        self.prepare_drift()
        requests = []
        files = {path.name: path.read_bytes() for path in self.code.iterdir()}

        def model(request):
            requests.append(request)
            assistants = [message for message in request.messages if message.tool_calls]
            bundles = [message for message in request.messages if message.tool_results]
            self.assertEqual(len(assistants), 1)
            self.assertEqual(len(bundles), 1)
            self.assertEqual(
                tuple(call.id for call in assistants[0].tool_calls), ("A", "B", "C")
            )
            self.assertEqual(
                tuple(block.tool_call_id for block in bundles[0].tool_results),
                ("A", "B", "C"),
            )
            self.assertFalse(bundles[0].tool_results[0].is_error)
            for block in bundles[0].tool_results[1:]:
                self.assertTrue(block.is_error)
            self.assertIn("请先重新读取", str(bundles[0].tool_results[1].content))
            self.assertIn(sha(self.current), str(bundles[0].tool_results[1].content))
            return response()

        with mock.patch(
            "horizon_harness.workspace.execution.atomic_edit_write",
            wraps=atomic_edit_write,
        ) as writes:
            adopted = self.adopt(note="reviewed")
            self.assertEqual(
                self.journal.batch_results(self.batch["batch_id"])[1]["is_error"], True
            )
            self.assertEqual(len(self.journal.batch_results(self.batch["batch_id"])), 3)
            self.session(model).resume("drift")
            self.session().resume("drift")
            writes.assert_not_called()
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.code.iterdir()}, files
        )
        self.assertEqual(adopted["sha256"], sha(self.current))
        payload = self.journal.step(self.step_id)["result_payload"]
        self.assertEqual(payload["reason"], "adopted_external_content")
        self.assertEqual(payload["note"], "reviewed")
        self.assertEqual(self.journal.step(self.step_id)["state"], "cancelled")
        self.assertEqual(
            sum(
                kind == "ToolBatchBundleCommitted"
                for _, kind, _ in self.journal.events()
            ),
            1,
        )
        self.assertEqual(self.journal.unsettled_step_ids(), ())

    def test_stale_expected_sha_refuses_without_state_change(self):
        self.prepare_drift()
        (self.code / "b").write_bytes(b"changed again")
        before = self.journal.events()
        step = self.journal.step(self.step_id)
        with self.assertRaisesRegex(ValueError, "expected-sha"):
            self.adopt()
        self.assertEqual(self.journal.events(), before)
        self.assertEqual(self.journal.step(self.step_id), step)
        self.assertEqual(len(self.journal.batch_results(self.batch["batch_id"])), 1)

    def test_non_drift_steps_are_rejected_without_state_change(self):
        self.prepare_drift()
        for index in (0, 2):
            with self.subTest(index=index):
                before = self.journal.events()
                with self.assertRaisesRegex(ValueError, "workspace drift"):
                    self.session().adopt_current(
                        "drift",
                        step_id=self.batch["calls"][index]["execution_step_id"],
                        expected_sha=sha(self.current),
                    )
                self.assertEqual(self.journal.events(), before)
        payload = self.journal.step(self.step_id)["result_payload"]
        for tool, reason in (("edit", "unsafe edit path"), ("bash", "workspace drift")):
            with self.subTest(tool=tool, reason=reason):
                self.journal.set_step_state(
                    self.step_id, "ambiguous", {**payload, "reason": reason}
                )
                with sqlite3.connect(self.journal.path) as db:
                    db.execute(
                        "UPDATE execution_steps SET tool_name=? WHERE step_id=?",
                        (tool, self.step_id),
                    )
                before = self.journal.events()
                with self.assertRaisesRegex(ValueError, "workspace drift"):
                    self.adopt()
                self.assertEqual(self.journal.events(), before)

    def test_unexecuted_suffix_is_cancelled_and_completed_steps_stay_unchanged(self):
        self.prepare_drift(
            extra_calls=(
                ToolCallBlock("D", "read", {"path": "a"}),
                ToolCallBlock("E", "read", {"path": "missing"}),
            )
        )
        completed = {}
        for index, state in ((3, "succeeded"), (4, "failed")):
            step_id = self.batch["calls"][index]["execution_step_id"]
            self.journal.commit_step_result(
                step_id,
                state=state,
                result_payload={
                    "tool_result": encode_tool_result(
                        text_result(state, is_error=state == "failed")
                    ),
                },
            )
            completed[step_id] = self.journal.step(step_id)
        self.adopt()
        cancelled = self.journal.step(self.batch["calls"][2]["execution_step_id"])
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertIn(
            "因前序编辑被取消而未执行，请重新规划",
            str(cancelled["result_payload"]["tool_result"]),
        )
        for step_id, step in completed.items():
            self.assertEqual(self.journal.step(step_id), step)
        self.assertEqual(
            tuple(
                row["call_id"]
                for row in self.journal.batch_results(self.batch["batch_id"])
            ),
            ("A", "B", "C", "D", "E"),
        )

    def test_suffix_with_execution_evidence_is_not_cancelled(self):
        self.prepare_drift()
        suffix_id = self.batch["calls"][2]["execution_step_id"]
        self.journal.commit_step_result(
            suffix_id, state="planned", result_payload={"execution_started": True}
        )
        before = self.journal.step(suffix_id)
        self.adopt()
        self.assertEqual(self.journal.step(suffix_id), before)
        self.assertEqual(len(self.journal.batch_results(self.batch["batch_id"])), 2)

    def test_suffix_start_event_prevents_cancellation_even_without_intent(self):
        self.prepare_drift()
        suffix_id = self.batch["calls"][2]["execution_step_id"]
        self.journal.append(
            "ToolExecutionStartEvent", {"tool_call_id": "C", "tool_name": "edit"}
        )
        before = self.journal.step(suffix_id)
        self.adopt()
        self.assertEqual(self.journal.step(suffix_id), before)

    def test_cancelled_suffix_results_remain_materializable_after_an_evidence_gap(self):
        self.prepare_drift(extra_calls=(ToolCallBlock("D", "read", {"path": "a"}),))
        suffix_id = self.batch["calls"][2]["execution_step_id"]
        self.journal.commit_step_result(
            suffix_id, state="planned", result_payload={"execution_started": True}
        )
        self.adopt()
        self.journal.commit_step_result(
            suffix_id,
            state="failed",
            result_payload={
                "tool_result": encode_tool_result(text_result("failed", is_error=True)),
            },
        )
        self.journal.materialize_terminal_batch_results(self.batch["batch_id"])
        results = self.journal.batch_results(self.batch["batch_id"])
        self.assertEqual(tuple(row["call_id"] for row in results), ("A", "B", "C", "D"))
        self.assertIn("请重新规划", str(results[-1]["tool_result"]))

    def test_repeated_adopt_is_rejected_without_duplicate_results(self):
        self.prepare_drift()
        self.adopt()
        before = self.journal.events()
        results = self.journal.batch_results(self.batch["batch_id"])
        with self.assertRaisesRegex(ValueError, "workspace drift"):
            self.adopt()
        self.assertEqual(self.journal.events(), before)
        self.assertEqual(self.journal.batch_results(self.batch["batch_id"]), results)

    def test_resume_without_adopt_remains_blocked_with_actionable_error(self):
        self.interrupt_edit()
        with mock.patch(
            "horizon_harness.workspace.execution.atomic_edit_write"
        ) as writes:
            for _ in range(2):
                with self.assertRaises(UnsafeCheckpointError) as error:
                    self.session(
                        lambda request: self.fail("unexpected model call")
                    ).resume("drift")
                for value in ("drift", self.step_id, "inspect", "adopt-current"):
                    self.assertIn(value, str(error.exception))
            writes.assert_not_called()
        self.assertEqual((self.code / "b").read_bytes(), self.current)

    def test_adopt_transaction_rolls_back_all_steps_and_results_on_failure(self):
        self.prepare_drift()
        before = self.journal.events()
        original = ExecutionJournal._append_step_event

        for failure_point in ("suffix", "result"):
            with self.subTest(failure_point=failure_point):

                def fail_commit(db, kind, payload):
                    if (
                        failure_point == "suffix"
                        and kind == "StepResultCommitted"
                        and payload["step_id"] != self.step_id
                    ) or (
                        failure_point == "result"
                        and kind == "ToolBatchResultCommitted"
                        and payload["call_index"] == 2
                    ):
                        raise RuntimeError("injected transaction failure")
                    original(db, kind, payload)

                with mock.patch.object(
                    ExecutionJournal, "_append_step_event", side_effect=fail_commit
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        self.adopt()
                self.assertEqual(self.journal.events(), before)
                self.assertEqual(self.journal.step(self.step_id)["state"], "ambiguous")
                self.assertEqual(
                    len(self.journal.batch_results(self.batch["batch_id"])), 1
                )

    def test_cli_inspect_adopt_and_resume(self):
        self.prepare_drift()
        args = ["--fake", "--code-dir", str(self.code), "--store-dir", str(self.store)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(args + ["inspect", "drift"]), 0)
        inspected = json.loads(output.getvalue())
        self.assertEqual(inspected["drift_reports"][0]["step_id"], self.step_id)
        self.assertEqual(
            inspected["drift_reports"][0]["current_sha256"], sha(self.current)
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                main(
                    args
                    + [
                        "adopt-current",
                        "drift",
                        "--step",
                        self.step_id,
                        "--expected-sha",
                        sha(self.current),
                        "--note",
                        "checked",
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(output.getvalue())["sha256"], sha(self.current))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(args + ["resume", "drift"]), 0)
        self.assertEqual(json.loads(output.getvalue())["task_id"], "drift")
        self.assertEqual((self.code / "b").read_bytes(), self.current)


if __name__ == "__main__":
    unittest.main()
