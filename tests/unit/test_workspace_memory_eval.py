from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from horizon_harness.llm import (
    FAKE_PROVIDER,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    llm_message,
)
from horizon_harness.messages import runtime_message
from horizon_harness.workspace import (
    Checkpoint,
    ModelWorkspaceProjector,
    TaskSession,
    WorkspaceStore,
)
from runs.workspace_memory_eval import (
    CELL_ORDER,
    MEMORY_SCOPE,
    BudgetExhausted,
    FrozenTask,
    TokenLedger,
    FormalRunner,
    _code_identity,
    _provider_fingerprint,
    _verifier_verdict,
    _verify,
    build_plan,
)


ROOT = Path(__file__).resolve().parents[2]
SELECTION = (
    ROOT / "evals/out/task-workspace-memory-delivery/swebench/task-selection.json"
)


class FormalWorkspaceMemoryPlanTests(unittest.TestCase):
    def test_prefix_rejects_uncommitted_checkpoint_before_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkspaceStore(root / "runs/prefix-H1/store")
            record = store.create("task")
            store.begin_turn(record.task_id, "run", 1)
            runner = object.__new__(FormalRunner)
            runner.output_dir = root
            task = FrozenTask(
                "H1", "case", "base", "image", root / "task.json", root / "oracle.json"
            )
            with mock.patch("runs.workspace_memory_eval._start_container") as start:
                with self.assertRaisesRegex(Exception, "uncommitted"):
                    runner._prefix(task)
                start.assert_not_called()

    def test_empty_patch_is_failure_not_infrastructure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(json.dumps({"empty_patch_ids": ["case"]}))
            self.assertEqual(_verifier_verdict("case", [report], 0), "fail")

    def test_setup_response_is_saved_and_output_budget_expanded(self) -> None:
        calls = []
        response = LLMResponse(usage=TokenUsage(input_tokens=3, output_tokens=2))

        def complete_fn(request):
            calls.append(request)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            ledger = TokenLedger(Path(tmp), complete_fn=complete_fn)
            ledger.complete("setup", "workspace-H1")(
                LLMRequest(provider=FAKE_PROVIDER, messages=[], max_tokens=2048)
            )
            self.assertEqual(calls[0].max_tokens, 8192)
            paths = list((Path(tmp) / "responses").glob("*.json"))
            self.assertEqual(len(paths), 1)
            self.assertEqual(
                json.loads(paths[0].read_text())["stop_reason"], "end_turn"
            )

    def test_plan_freezes_exact_matrix_and_keeps_oracle_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan(SELECTION, Path(tmp), implementation=ROOT)

            self.assertEqual(
                [row["cell"] for row in plan["suffix_order"]],
                [row[0] for row in CELL_ORDER],
            )
            self.assertEqual(
                {row["role"] for row in plan["tasks"]}, {"D", "H1", "H2", "H3"}
            )
            self.assertEqual(plan["memory_scope"], MEMORY_SCOPE)
            for row in plan["tasks"]:
                task = json.loads(Path(row["task_file"]).read_text(encoding="utf-8"))
                self.assertFalse(
                    {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"} & set(task)
                )
                self.assertNotEqual(
                    Path(row["task_file"]).parent,
                    Path(row["verifier_file"]).parent,
                )

    def test_budget_reserves_before_call_and_reconciles_known_usage(self) -> None:
        response = LLMResponse(
            content=(),
            usage=TokenUsage(input_tokens=11, output_tokens=7),
            model="fake-model",
        )
        called = []

        def complete_fn(request):
            called.append(request)
            return response

        with tempfile.TemporaryDirectory() as tmp:
            ledger = TokenLedger(Path(tmp), total_limit=10_000, complete_fn=complete_fn)
            request = LLMRequest(
                provider=FAKE_PROVIDER,
                messages=[llm_message("user", "hello")],
                max_tokens=100,
            )
            actual = ledger.complete("development", "unit")(request)
            state = json.loads((Path(tmp) / "budget.json").read_text(encoding="utf-8"))
            usage = [
                json.loads(line)
                for line in (Path(tmp) / "usage.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertIs(actual, response)
            self.assertEqual(len(called), 1)
            self.assertEqual(state["charged"], 18)
            self.assertEqual(usage[0]["charged"], 18)
            self.assertEqual(usage[0]["status"], "ok")

    def test_budget_refuses_call_when_reservation_exceeds_total(self) -> None:
        called = []

        def complete_fn(request):
            called.append(request)
            return LLMResponse()

        with tempfile.TemporaryDirectory() as tmp:
            ledger = TokenLedger(Path(tmp), total_limit=1, complete_fn=complete_fn)
            request = LLMRequest(
                provider=FAKE_PROVIDER,
                messages=[llm_message("user", "hello")],
                max_tokens=100,
            )
            with self.assertRaises(BudgetExhausted):
                ledger.complete("development", "unit")(request)
            self.assertEqual(called, [])

    def test_official_report_resolved_id_is_a_pass(self) -> None:
        report = (
            ROOT / "evals/out/task-workspace-memory-delivery/swebench/"
            "verifier-smoke-D/verifier/"
            "horizon-workspace-memory.verifier-smoke-D.json"
        )
        self.assertEqual(
            _verifier_verdict("pytest-dev__pytest-11143", [report], 0), "pass"
        )
        self.assertEqual(
            _verifier_verdict("pytest-dev__pytest-missing", [report], 0),
            "infra_error",
        )

    def test_verifier_timeout_is_recorded_without_aborting(self) -> None:
        task = FrozenTask(
            role="D",
            instance_id="pytest-dev__pytest-11143",
            base_commit="base",
            image="image",
            task_file=Path("task.json"),
            verifier_file=Path("verifier.jsonl"),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch(
                "runs.workspace_memory_eval.subprocess.run",
                side_effect=subprocess.TimeoutExpired(
                    ["swebench"], 660, output="partial", stderr="timed out"
                ),
            ),
        ):
            verdict, result = _verify(task, "patch", Path(tmp))

            self.assertEqual(verdict, "timeout")
            self.assertEqual(
                json.loads(result.read_text(encoding="utf-8"))["verdict"],
                "timeout",
            )
            self.assertEqual(
                (result.parent / "stdout.log").read_text(encoding="utf-8"),
                "partial",
            )

    def test_workspace_projection_accepts_json_fence_and_forces_pause(self) -> None:
        response = LLMResponse(
            content=(
                runtime_message(
                    """```json
{"steps":[{"id":"inspect","description":"Inspect code","status":"completed"}],"artifacts":[{"path":"result.txt","step_ids":["inspect"]}],"next_action":"Run tests","status":"completed"}
```"""
                ).content[0],
            ),
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            model="fake-model",
        )
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp) / "code"
            code.mkdir()
            (code / "result.txt").write_text("ok", encoding="utf-8")
            record = WorkspaceStore(Path(tmp) / "store").create("fix it")

            projected = ModelWorkspaceProjector(
                FAKE_PROVIDER, complete_fn=lambda request: response
            ).project(record, [runtime_message("inspected")], code_dir=code)

            self.assertEqual(projected.record.status, "paused")
            self.assertEqual(projected.record.steps[0].status, "done")
            self.assertEqual(projected.record.next_action, "Run tests")
            self.assertEqual(projected.record.artifacts[0].path, "result.txt")

    def test_workspace_projection_rejects_missing_artifact(self) -> None:
        response = LLMResponse(
            content=(
                runtime_message('{"artifacts":[{"path":"missing.txt"}]}').content[0],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp) / "code"
            code.mkdir()
            record = WorkspaceStore(Path(tmp) / "store").create("fix it")

            with self.assertRaisesRegex(ValueError, "does not exist"):
                ModelWorkspaceProjector(
                    FAKE_PROVIDER, complete_fn=lambda request: response
                ).project(record, [runtime_message("inspected")], code_dir=code)

    def test_external_projection_artifact_is_explicitly_not_persisted(self) -> None:
        response = LLMResponse(
            content=runtime_message('{"artifacts":[{"path":"/tmp/repro.py"}]}').content
        )
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp)
            record = WorkspaceStore(code / "store").create("fix it")
            result = ModelWorkspaceProjector(
                FAKE_PROVIDER, complete_fn=lambda request: response
            ).project(record, [], code_dir=code)
            self.assertEqual(result.record.artifacts, ())
            self.assertIn("/tmp/repro.py", result.record.blockers[0])

    def test_plain_prefix_projection_resumes_with_matching_fingerprint(self) -> None:
        response = LLMResponse(
            content=(
                runtime_message(
                    '{"steps":[{"id":"inspect","description":"Inspect code",'
                    '"status":"done"}],"next_action":"Continue"}'
                ).content[0],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp) / "code"
            code.mkdir()
            store = WorkspaceStore(Path(tmp) / "store")
            record = store.create("finish it")
            record = replace(
                record,
                status="paused",
                checkpoint=Checkpoint(
                    run_id="prefix",
                    completed_turn=1,
                    resumable=True,
                    provider_fingerprint=_provider_fingerprint(FAKE_PROVIDER),
                    code_identity=_code_identity(code),
                ),
            )
            record = store.commit_checkpoint(
                record, [runtime_message("prefix evidence")], []
            )
            projected = ModelWorkspaceProjector(
                FAKE_PROVIDER, complete_fn=lambda request: response
            ).project(record, store.load_messages(record), code_dir=code)
            store.save(projected.record)

            resumed = TaskSession(
                FAKE_PROVIDER, code_dir=code, store_dir=store.root
            ).resume(record.task_id, pause_after_turns=1)

            self.assertEqual(resumed.status, "paused")
            self.assertGreater(len(resumed.state.messages), 1)


if __name__ == "__main__":
    unittest.main()
