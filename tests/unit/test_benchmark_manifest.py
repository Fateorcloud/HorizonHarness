from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from horizon_harness.evals.benchmark import (
    API_PREFLIGHT_CASES,
    FAILURE_TYPES,
    MODEL_MATRIX,
    RELIABILITY_CASES,
    REPORT_METRICS,
    SWE_BENCH_PRO_TASK_IDS,
    TERMINAL_BENCH_PILOT_TASK_IDS,
    build_default_manifest,
    build_benchmark_report,
    manifest_to_dict,
    compare_reports,
    validate_offline_paths,
    validate_report,
    validate_manifest,
)


class BenchmarkManifestTests(unittest.TestCase):
    def test_default_manifest_freezes_three_models_and_runtime_axes(self) -> None:
        manifest = build_default_manifest(commit="abc123")
        payload = manifest_to_dict(manifest)

        self.assertEqual(payload["schema"], "horizon-harness.benchmark.v1")
        self.assertEqual(payload["commit"], "abc123")
        self.assertEqual(
            [model["model"] for model in payload["models"]],
            ["gpt-5.6-sol", "z-ai/glm-5.2:free", "deepseek-v4-flash"],
        )
        self.assertEqual(payload["agent"]["capabilities"], ["bash", "read", "task"])
        self.assertEqual(payload["agent"]["workflow"], "single")
        self.assertEqual(payload["agent"]["memory"], False)
        self.assertEqual(payload["agent"]["skills"], False)
        self.assertEqual(payload["agent"]["mcp"], False)

    def test_stage_case_counts_and_budgets_are_explicit(self) -> None:
        stages = manifest_to_dict(build_default_manifest())["stages"]
        self.assertEqual(stages["api_preflight"]["cases"], 5)
        self.assertEqual(stages["runtime_reliability"]["cases"], 18)
        self.assertEqual(stages["terminal_bench"]["pilot_cases"], 20)
        self.assertEqual(stages["terminal_bench"]["max_turns"], 150)
        self.assertEqual(stages["swe_bench_pro"]["cases"], 25)
        self.assertEqual(stages["swe_bench_pro"]["max_turns"], 200)

    def test_model_matrix_has_only_supported_provider_apis(self) -> None:
        self.assertEqual(
            [(m.model, m.api, m.reasoning) for m in MODEL_MATRIX],
            [
                ("gpt-5.6-sol", "openai-responses", "high"),
                ("z-ai/glm-5.2:free", "openai-chat", None),
                ("deepseek-v4-flash", "openai-chat", None),
            ],
        )
        self.assertEqual(
            set(FAILURE_TYPES),
            {
                "model_decision",
                "tool_input",
                "tool_execution",
                "provider_adapter",
                "runtime",
                "budget_exhausted",
                "task_incomplete",
                "scorer_or_infra",
            },
        )

    def test_validation_rejects_matrix_or_runtime_drift(self) -> None:
        payload = manifest_to_dict(build_default_manifest())
        payload["models"][0]["api"] = "anthropic-messages"
        with self.assertRaisesRegex(ValueError, "model matrix"):
            validate_manifest(payload)

    def test_run_manifest_freezes_reproducibility_fields(self) -> None:
        from horizon_harness.evals.benchmark import build_run_manifest

        payload = build_run_manifest(
            run_id="tb21-glm-pilot",
            commit="abc123",
            dirty=True,
            benchmark={"name": "terminal-bench", "version": "2.1"},
            task_selection={"task_ids": ["t1", "t2"]},
            model={
                "requested_id": "z-ai/glm-5.2:free",
                "api": "openai-chat",
                "provider": "openrouter",
            },
            agent={"capabilities": ["bash", "read", "task"], "workflow": "single"},
            budget={"max_turns": 150, "max_output_tokens": 4096},
            runtime={
                "docker_image": "terminal-bench@sha256:deadbeef",
                "scorer_version": "terminal-bench-scorer-2.1",
            },
            artifact_root="evals/out/harbor/jobs/tb21-glm-pilot",
            run_at="2026-09-01T00:00:00+00:00",
        )
        self.assertEqual(payload["schema"], "horizon-harness.run-manifest.v1")
        self.assertEqual(payload["run_id"], "tb21-glm-pilot")
        self.assertEqual(payload["source"], {"git_commit": "abc123", "git_dirty": True})
        self.assertEqual(payload["model"]["requested_id"], "z-ai/glm-5.2:free")
        self.assertEqual(
            payload["runtime"]["docker_image"],
            "terminal-bench@sha256:deadbeef",
        )
        self.assertEqual(len(payload["task_selection"]["sha256"]), 64)

        reordered = build_run_manifest(
            run_id="tb21-glm-pilot",
            commit="abc123",
            dirty=True,
            benchmark={"name": "terminal-bench", "version": "2.1"},
            task_selection={"task_ids": ["t1", "t2"]},
            model={"requested_id": "z-ai/glm-5.2:free"},
            agent={"workflow": "single"},
            budget={"max_turns": 150},
            runtime={},
            artifact_root="out",
            run_at="2026-09-01T00:00:00+00:00",
        )
        self.assertEqual(
            reordered["task_selection"]["sha256"],
            payload["task_selection"]["sha256"],
        )

        payload = manifest_to_dict(build_default_manifest())
        payload["agent"]["workflow"] = "planner_executor"
        with self.assertRaisesRegex(ValueError, "single"):
            validate_manifest(payload)

    def test_frozen_case_lists_are_ordered_and_unique(self) -> None:
        payload = manifest_to_dict(build_default_manifest())
        stages = payload["stages"]
        self.assertEqual(
            tuple(stages["api_preflight"]["case_types"]), API_PREFLIGHT_CASES
        )
        self.assertEqual(len(set(RELIABILITY_CASES)), 18)
        self.assertEqual(len(set(TERMINAL_BENCH_PILOT_TASK_IDS)), 20)
        pilot_file = (
            Path(__file__).resolve().parents[2]
            / "runs/harbor/terminal_bench_2_1_pilot.txt"
        )
        pilot_names = tuple(pilot_file.read_text(encoding="utf-8").splitlines())
        self.assertEqual(pilot_names, TERMINAL_BENCH_PILOT_TASK_IDS)
        self.assertEqual(
            hashlib.sha256(("\n".join(pilot_names) + "\n").encode()).hexdigest(),
            "e9312453dad7028c1381d4dd7131bfe0fb49329d610fb533da19aee0e7d3f328",
        )
        self.assertEqual(len(SWE_BENCH_PRO_TASK_IDS), 25)
        self.assertEqual(len(set(SWE_BENCH_PRO_TASK_IDS)), 25)
        self.assertTrue(
            all(value.startswith("instance_") for value in SWE_BENCH_PRO_TASK_IDS)
        )

    def test_report_requires_all_metrics(self) -> None:
        metrics = {name: 0 for name in REPORT_METRICS}
        report = build_benchmark_report(
            benchmark="terminal-bench",
            benchmark_version="2.1",
            model=MODEL_MATRIX[0],
            harness_commit="abc",
            metrics=metrics,
        )
        validate_report(report)
        incomplete = dict(report)
        incomplete["metrics"] = {"score": 1}
        with self.assertRaisesRegex(ValueError, "missing metrics"):
            validate_report(incomplete)

    def test_offline_fake_and_oracle_gate_paid_runs(self) -> None:
        ready = validate_offline_paths(
            fake_runner=lambda: {"trace_complete": True, "artifact_complete": True},
            oracle_runner=lambda: {
                "trace_complete": True,
                "artifact_complete": True,
                "oracle_score_ok": True,
            },
        )
        self.assertTrue(ready["ready_for_paid_run"])
        with self.assertRaisesRegex(ValueError, "complete trace"):
            validate_offline_paths(
                fake_runner=lambda: {
                    "trace_complete": False,
                    "artifact_complete": True,
                },
                oracle_runner=lambda: {
                    "trace_complete": True,
                    "artifact_complete": True,
                    "oracle_score_ok": True,
                },
            )
        with self.assertRaisesRegex(ValueError, "expected score"):
            validate_offline_paths(
                fake_runner=lambda: {"trace_complete": True, "artifact_complete": True},
                oracle_runner=lambda: {
                    "trace_complete": True,
                    "artifact_complete": True,
                    "oracle_score_ok": False,
                },
            )

    def test_comparison_marks_different_identity_as_public_reference(self) -> None:
        metrics = {name: 0 for name in REPORT_METRICS}
        baseline = build_benchmark_report(
            benchmark="terminal-bench",
            benchmark_version="2.1",
            model=MODEL_MATRIX[0],
            harness_commit="abc",
            metrics={**metrics, "score": 0.4},
        )
        candidate = build_benchmark_report(
            benchmark="terminal-bench",
            benchmark_version="2.1",
            model=MODEL_MATRIX[1],
            harness_commit="def",
            metrics={**metrics, "score": 0.8},
        )
        reference = compare_reports(baseline, candidate)
        self.assertFalse(reference["comparable"])
        self.assertTrue(reference["public_reference"])
        self.assertIsNone(reference["delta"])

        same = build_benchmark_report(
            benchmark="terminal-bench",
            benchmark_version="2.1",
            model=MODEL_MATRIX[0],
            harness_commit="abc",
            metrics={**metrics, "score": 0.7},
        )
        self.assertEqual(compare_reports(baseline, same)["delta"], 0.3)


if __name__ == "__main__":
    unittest.main()
