"""Public TaskSession crash contracts; no handcrafted recovery database."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from horizon_harness.llm.env import FAKE_PROVIDER
from horizon_harness.llm.types import LLMResponse
from horizon_harness.messages import TextBlock, ToolCallBlock
from horizon_harness.workspace import TaskSession


def three_edit_response() -> LLMResponse:
    return LLMResponse(
        content=tuple(
            ToolCallBlock(
                call_id,
                "edit",
                {
                    "path": f"{call_id.lower()}.txt",
                    "old_string": "0",
                    "new_string": replacement,
                },
            )
            for call_id, replacement in (("A", "1"), ("B", "2"), ("C", "3"))
        ),
        stop_reason="tool_use",
        model="fake-model",
    )


def crash_after_decision(root: Path, store: Path) -> None:
    from horizon_harness.workspace.journal import ExecutionJournal

    commit = ExecutionJournal.commit_model_decision

    def commit_then_exit(self, **kwargs):
        result = commit(self, **kwargs)
        if kwargs["batch_id"]:
            os._exit(91)
        return result

    ExecutionJournal.commit_model_decision = commit_then_exit

    def model(request):
        return three_edit_response()

    TaskSession(FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=model).start(
        "Replace before with after", task_id="crash-A"
    )


def crash_after_two_call_decision(root: Path, store: Path) -> None:
    from horizon_harness.workspace.journal import ExecutionJournal

    commit = ExecutionJournal.commit_model_decision

    def commit_then_exit(self, **kwargs):
        result = commit(self, **kwargs)
        if kwargs["batch_id"]:
            os._exit(91)
        return result

    ExecutionJournal.commit_model_decision = commit_then_exit

    def model(request):
        return three_edit_response()

    TaskSession(FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=model).start(
        "Apply two ordered edits", task_id="crash-two"
    )


def crash_after_first_result(root: Path, store: Path) -> None:
    from horizon_harness.workspace.journal import ExecutionJournal

    record = ExecutionJournal.record_batch_result

    def record_then_exit(self, batch_id, call_index, result, **kwargs):
        value = record(self, batch_id, call_index, result, **kwargs)
        if call_index == 0:
            os._exit(91)
        return value

    ExecutionJournal.record_batch_result = record_then_exit

    def model(request):
        return three_edit_response()

    TaskSession(FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=model).start(
        "Apply two ordered edits", task_id="crash-B"
    )


def crash_during_recovery_after_first_result(root: Path, store: Path) -> None:
    from horizon_harness.workspace.journal import ExecutionJournal

    record = ExecutionJournal.record_batch_result

    def record_then_exit(self, batch_id, call_index, result, **kwargs):
        value = record(self, batch_id, call_index, result, **kwargs)
        if call_index == 0:
            os._exit(91)
        return value

    ExecutionJournal.record_batch_result = record_then_exit

    def model(request):
        raise AssertionError("recovery crash must happen before the next model call")

    TaskSession(
        FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=model
    ).resume("crash-A")


def crash_before_bundle_seal(root: Path, store: Path) -> None:
    from horizon_harness.workspace.journal import ExecutionJournal

    def seal_then_exit(self, batch_id, *args, **kwargs):
        os._exit(91)

    ExecutionJournal.commit_batch_bundle = seal_then_exit

    def model(request):
        return three_edit_response()

    TaskSession(FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=model).start(
        "Apply one edit", task_id="crash-D"
    )


def crash_after_edit_effect(root: Path, store: Path) -> None:
    os.environ["HORIZON_CRASH_AFTER_EDIT_INDEX"] = "1"

    def model(request):
        return three_edit_response()

    TaskSession(FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=model).start(
        "Apply one edit", task_id="crash-C"
    )


def crash_after_bundle_seal(root: Path, store: Path) -> None:
    from horizon_harness.workspace.journal import ExecutionJournal

    commit = ExecutionJournal.commit_batch_bundle

    def commit_then_exit(self, batch_id, message_event):
        result = commit(self, batch_id, message_event)
        os._exit(91)
        return result

    ExecutionJournal.commit_batch_bundle = commit_then_exit

    def model(request):
        return three_edit_response()

    TaskSession(FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=model).start(
        "Apply one edit", task_id="crash-E"
    )


class RuntimeCrashContractTests(unittest.TestCase):
    def _prepare_code(self, directory: str) -> Path:
        root = Path(directory) / "code"
        root.mkdir()
        for name in ("a", "b", "c"):
            (root / f"{name}.txt").write_text("0", encoding="utf-8")
        return root

    def _assert_applied(self, root: Path) -> None:
        self.assertEqual(
            tuple((root / f"{name}.txt").read_text() for name in ("a", "b", "c")),
            ("1", "2", "3"),
        )

    def _assert_provider_history(self, request) -> None:
        assistant = [message for message in request.messages if message.tool_calls]
        results = [message for message in request.messages if message.tool_results]
        self.assertEqual(len(assistant), 1)
        self.assertEqual(
            tuple(call.id for call in assistant[0].tool_calls), ("A", "B", "C")
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(
            tuple(block.tool_call_id for block in results[0].tool_results),
            ("A", "B", "C"),
        )

    def test_C_public_resume_reconciles_edit_effect_without_duplicate_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._prepare_code(directory)
            store = Path(directory) / "store"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.unit.test_runtime_crash_contract",
                    str(root),
                    str(store),
                    "C",
                ],
                timeout=30,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 91, process.stderr)
            calls = []

            def next_model(request):
                calls.append(request)
                self._assert_applied(root)
                self._assert_provider_history(request)
                return LLMResponse(
                    content=(TextBlock("done"),),
                    stop_reason="end_turn",
                    model="fake-model",
                )

            TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=next_model
            ).resume("crash-C")
            self.assertEqual(len(calls), 1)

    def test_A_committed_decision_resumes_without_regenerating_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._prepare_code(directory)
            store = Path(directory) / "store"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.unit.test_runtime_crash_contract",
                    str(root),
                    str(store),
                ],
                timeout=30,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 91, process.stderr)
            self.assertEqual(
                tuple((root / f"{name}.txt").read_text() for name in ("a", "b", "c")),
                ("0", "0", "0"),
            )
            calls = []

            def next_model(request):
                calls.append(request)
                self._assert_applied(root)
                self._assert_provider_history(request)
                return LLMResponse(
                    content=(TextBlock("done"),),
                    stop_reason="end_turn",
                    model="fake-model",
                )

            TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=next_model
            ).resume("crash-A")
            self.assertEqual(len(calls), 1)

    def test_A_three_call_batch_preserves_order_and_ids_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._prepare_code(directory)
            store = Path(directory) / "store"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.unit.test_runtime_crash_contract",
                    str(root),
                    str(store),
                    "two",
                ],
                timeout=30,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 91, process.stderr)
            calls = []

            def next_model(request):
                calls.append(request)
                self._assert_applied(root)
                self._assert_provider_history(request)
                return LLMResponse(
                    content=(TextBlock("done"),),
                    stop_reason="end_turn",
                    model="fake-model",
                )

            TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=next_model
            ).resume("crash-two")
            self.assertEqual(len(calls), 1)

    def test_D_bundle_commit_is_atomic_by_batch_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._prepare_code(directory)
            store = Path(directory) / "store"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.unit.test_runtime_crash_contract",
                    str(root),
                    str(store),
                    "D",
                ],
                timeout=30,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 91, process.stderr)
            from horizon_harness.workspace.journal import ExecutionJournal

            journal = ExecutionJournal(store / "crash-D" / "journal.sqlite3")
            before_resume = journal.rebuild_messages()
            self.assertEqual(
                sum(message.kind == "tool_result" for message in before_resume), 0
            )
            calls = []

            def next_model(request):
                calls.append(request)
                self._assert_applied(root)
                self._assert_provider_history(request)
                return LLMResponse(
                    content=(TextBlock("done"),),
                    stop_reason="end_turn",
                    model="fake-model",
                )

            TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=next_model
            ).resume("crash-D")
            self.assertEqual(len(calls), 1)

    def test_E_sealed_bundle_is_not_duplicated_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._prepare_code(directory)
            store = Path(directory) / "store"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.unit.test_runtime_crash_contract",
                    str(root),
                    str(store),
                    "E",
                ],
                timeout=30,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 91, process.stderr)
            calls = []

            def next_model(request):
                calls.append(request)
                self._assert_applied(root)
                self._assert_provider_history(request)
                return LLMResponse(
                    content=(TextBlock("done"),),
                    stop_reason="end_turn",
                    model="fake-model",
                )

            result = TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=next_model
            ).resume("crash-E")
            self.assertEqual(len(calls), 1)
            messages = result.state.messages
            self.assertEqual(
                sum(
                    message.role == "assistant" and bool(message.tool_calls)
                    for message in messages
                ),
                1,
            )
            self.assertEqual(
                sum(
                    any(
                        getattr(block, "kind", "") == "tool_result"
                        for block in message.content
                    )
                    for message in messages
                ),
                1,
            )

    def test_B_committed_first_tool_is_not_reexecuted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._prepare_code(directory)
            store = Path(directory) / "store"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.unit.test_runtime_crash_contract",
                    str(root),
                    str(store),
                    "B",
                ],
                timeout=30,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 91, process.stderr)
            calls = []

            def next_model(request):
                calls.append(request)
                self._assert_applied(root)
                self._assert_provider_history(request)
                return LLMResponse(
                    content=(TextBlock("done"),),
                    stop_reason="end_turn",
                    model="fake-model",
                )

            TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=next_model
            ).resume("crash-B")
            self.assertEqual(len(calls), 1)

    def test_recovery_can_crash_again_and_resume_the_same_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._prepare_code(directory)
            store = Path(directory) / "store"
            first = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.unit.test_runtime_crash_contract",
                    str(root),
                    str(store),
                ],
                timeout=30,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 91, first.stderr)
            second = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.unit.test_runtime_crash_contract",
                    str(root),
                    str(store),
                    "resume-B",
                ],
                timeout=30,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 91, second.stderr)
            self.assertEqual(
                tuple((root / f"{name}.txt").read_text() for name in ("a", "b", "c")),
                ("1", "0", "0"),
            )
            calls = []

            def next_model(request):
                calls.append(request)
                self._assert_applied(root)
                self._assert_provider_history(request)
                return LLMResponse(
                    content=(TextBlock("done"),),
                    stop_reason="end_turn",
                    model="fake-model",
                )

            TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=store, complete_fn=next_model
            ).resume("crash-A")
            self.assertEqual(len(calls), 1)
            from horizon_harness.workspace.journal import ExecutionJournal

            steps = ExecutionJournal(
                store / "crash-A" / "journal.sqlite3"
            ).rebuild_steps()
            self.assertEqual(
                tuple(step["attempt"] for step in steps.values()), (1, 1, 1)
            )


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "C":
        crash_after_edit_effect(Path(sys.argv[1]), Path(sys.argv[2]))
    elif len(sys.argv) > 3 and sys.argv[3] == "E":
        crash_after_bundle_seal(Path(sys.argv[1]), Path(sys.argv[2]))
    elif len(sys.argv) > 3 and sys.argv[3] == "D":
        crash_before_bundle_seal(Path(sys.argv[1]), Path(sys.argv[2]))
    elif len(sys.argv) > 3 and sys.argv[3] == "B":
        crash_after_first_result(Path(sys.argv[1]), Path(sys.argv[2]))
    elif len(sys.argv) > 3 and sys.argv[3] == "resume-B":
        crash_during_recovery_after_first_result(Path(sys.argv[1]), Path(sys.argv[2]))
    elif len(sys.argv) > 3 and sys.argv[3] == "two":
        crash_after_two_call_decision(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        crash_after_decision(Path(sys.argv[1]), Path(sys.argv[2]))
