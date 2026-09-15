"""Offline benchmark matrix and run-manifest contracts.

The manifest is deliberately data-only.  It freezes the model/API pair,
runtime axes, stage budgets, and failure taxonomy before any paid benchmark
run starts.  It never reads credentials or starts a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

FAILURE_TYPES: tuple[str, ...] = (
    "model_decision",
    "tool_input",
    "tool_execution",
    "provider_adapter",
    "runtime",
    "budget_exhausted",
    "task_incomplete",
    "scorer_or_infra",
)

API_PREFLIGHT_CASES: tuple[str, ...] = (
    "text",
    "single_tool",
    "multi_tool",
    "error_tool",
    "long_output",
)

RELIABILITY_CASES: tuple[str, ...] = (
    "tool_arguments_valid",
    "tool_arguments_missing",
    "tool_arguments_extra",
    "tool_call_order",
    "tool_calls_parallel",
    "tool_failure_returned",
    "tool_timeout_returned",
    "provider_error_returned",
    "max_turns_terminates",
    "explicit_stop_terminates",
    "context_view_projection",
    "context_compression_trigger",
    "context_compression_preserves_task",
    "trace_header_written",
    "trace_events_indexed",
    "trace_artifact_complete",
    "result_artifact_complete",
    "budget_exhaustion_classified",
)

# Stable sample from the 89 official Terminal-Bench 2.1 task names: sort the
# names by SHA-256(name), take the first 20, then store them alphabetically for
# review. The newline-delimited list hashes to
# e9312453dad7028c1381d4dd7131bfe0fb49329d610fb533da19aee0e7d3f328.
TERMINAL_BENCH_PILOT_TASK_IDS: tuple[str, ...] = (
    "adaptive-rejection-sampler",
    "build-pmars",
    "code-from-image",
    "extract-elf",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "fix-git",
    "gcode-to-text",
    "gpt2-codegolf",
    "headless-terminal",
    "hf-model-inference",
    "make-mips-interpreter",
    "password-recovery",
    "pytorch-model-recovery",
    "regex-log",
    "sparql-university",
    "sqlite-db-truncate",
    "torch-pipeline-parallelism",
    "vulnerable-secret",
    "write-compressor",
)

# A fixed, ordered auxiliary set. These are opaque task IDs so the manifest can
# be reviewed and compared without bundling the benchmark's private prompts.
SWE_BENCH_PRO_TASK_IDS: tuple[str, ...] = (
    "instance_NodeBB__NodeBB-0e07f3c9bace416cbab078a30eae972868c0a8a3-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e",
    "instance_NodeBB__NodeBB-84e065752f6d7fbe5c08cbf50cb173ffb866b8fa-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e",
    "instance_NodeBB__NodeBB-0c81642997ea1d827dbd02c311db9d4976112cd4-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e",
    "instance_NodeBB__NodeBB-087e6020e490b4a1759f38c1ad03869511928263-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e",
    "instance_NodeBB__NodeBB-9c576a0758690f45a6ca03b5884c601e473bf2c1-vd59a5728dfc977f44533186ace531248c2917516",
    "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan",
    "instance_NodeBB__NodeBB-6ea3b51f128dd270281db576a1b59270d5e45db0-vnan",
    "instance_NodeBB__NodeBB-f083cd559d69c16481376868c8da65172729c0ca-vnan",
    "instance_NodeBB__NodeBB-70b4a0e2aebebe8f2f559de6680093d96a697b2f-vnan",
    "instance_NodeBB__NodeBB-b1f9ad5534bb3a44dab5364f659876a4b7fe34c1-vnan",
    "instance_ansible__ansible-164881d871964aa64e0f911d03ae270acbad253c-v390e508d27db7a51eece36bb6d9698b63a5b638a",
    "instance_ansible__ansible-ecea15c508f0e081525be036cf76bbb56dbcdd9d-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_ansible__ansible-ea04e0048dbb3b63f876aad7020e1de8eee9f362-v1055803c3a812189a1133297f7f5468579283f86",
    "instance_ansible__ansible-984216f52e76b904e5b0fa0fb956ab4f1e0a7751-v1055803c3a812189a1133297f7f5468579283f86",
    "instance_ansible__ansible-fb144c44144f8bd3542e71f5db62b6d322c7bd85-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_ansible__ansible-83909bfa22573777e3db5688773bda59721962ad-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_ansible__ansible-de5858f48dc9e1ce9117034e0d7e76806f420ca8-v1055803c3a812189a1133297f7f5468579283f86",
    "instance_ansible__ansible-f8ef34672b961a95ec7282643679492862c688ec-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_ansible__ansible-be2c376ab87e3e872ca21697508f12c6909cf85a-v1055803c3a812189a1133297f7f5468579283f86",
    "instance_ansible__ansible-9759e0ca494de1fd5fc2df2c5d11c57adbe6007c-v1055803c3a812189a1133297f7f5468579283f86",
    "instance_ansible__ansible-f327e65d11bb905ed9f15996024f857a95592629-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_ansible__ansible-935528e22e5283ee3f63a8772830d3d01f55ed8c-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_ansible__ansible-29aea9ff3466e4cd2ed00524b9e56738d568ce8b-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5",
    "instance_ansible__ansible-ed6581e4db2f1bec5a772213c3e186081adc162d-v0f01c69f1e2528b935359cfe578530722bca2c59",
    "instance_ansible__ansible-5e369604e1930b1a2e071fecd7ec5276ebd12cb1-v0f01c69f1e2528b935359cfe578530722bca2c59",
)

REPORT_METRICS: tuple[str, ...] = (
    "score",
    "run_success_rate",
    "tool_error_rate",
    "timeout_rate",
    "budget_exhaustion_rate",
    "trace_complete_rate",
    "model_calls",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "elapsed_seconds",
    "cost_usd",
)


@dataclass(frozen=True)
class ModelBenchmarkConfig:
    model: str
    api: str
    reasoning: str | None = None


MODEL_MATRIX: tuple[ModelBenchmarkConfig, ...] = (
    ModelBenchmarkConfig("gpt-5.6-sol", "openai-responses", "high"),
    ModelBenchmarkConfig("z-ai/glm-5.2:free", "openai-chat"),
    ModelBenchmarkConfig("deepseek-v4-flash", "openai-chat"),
)


@dataclass(frozen=True)
class BenchmarkAgentConfig:
    capabilities: tuple[str, ...] = ("bash", "read", "task")
    workflow: str = "single"
    memory: bool = False
    skills: bool = False
    mcp: bool = False
    compression_ratio: float = 0.8


@dataclass(frozen=True)
class BenchmarkManifest:
    schema: str = "horizon-harness.benchmark.v1"
    commit: str | None = None
    models: tuple[ModelBenchmarkConfig, ...] = MODEL_MATRIX
    agent: BenchmarkAgentConfig = BenchmarkAgentConfig()
    stages: Mapping[str, Mapping[str, Any]] | None = None
    failure_types: tuple[str, ...] = FAILURE_TYPES
    task_sets: Mapping[str, Mapping[str, Any]] | None = None
    budget: Mapping[str, Any] | None = None
    docker_image: str = ""
    scorer_version: str = ""
    price_source: str = ""
    run_at: str = ""

    def __post_init__(self) -> None:
        if self.stages is None:
            object.__setattr__(self, "stages", _default_stages())
        if self.task_sets is None:
            object.__setattr__(self, "task_sets", _default_task_sets())
        if self.budget is None:
            object.__setattr__(
                self,
                "budget",
                {
                    "max_turns": {"terminal_bench": 150, "swe_bench_pro": 200},
                    "paid_runs_allowed": False,
                },
            )


def _default_stages() -> dict[str, dict[str, Any]]:
    return {
        "api_preflight": {
            "cases": 5,
            "case_types": list(API_PREFLIGHT_CASES),
            "requires_all_models": True,
        },
        "runtime_reliability": {
            "cases": 18,
            "requires_all_models": True,
            "trace_complete_rate_target": 1.0,
            "backend": "fake",
            "case_ids": list(RELIABILITY_CASES),
        },
        "terminal_bench": {
            "version": "2.1",
            "pilot_cases": 20,
            "pilot_task_ids": list(TERMINAL_BENCH_PILOT_TASK_IDS),
            "official_task_set": True,
            "max_turns": 150,
            "requires_all_models": True,
        },
        "swe_bench_pro": {
            "version": "pro",
            "cases": 25,
            "task_ids": list(SWE_BENCH_PRO_TASK_IDS),
            "fixed_subset": True,
            "max_turns": 200,
            "requires_all_models": True,
        },
    }


def _default_task_sets() -> dict[str, dict[str, Any]]:
    """Describe frozen task-set inputs without downloading benchmark data."""

    return {
        "terminal_bench": {
            "name": "Terminal-Bench 2.1",
            "version": "2.1",
            "pilot_cases": 20,
            "pilot_task_ids": list(TERMINAL_BENCH_PILOT_TASK_IDS),
            "official_task_set": True,
        },
        "swe_bench_pro": {
            "name": "SWE-bench Pro",
            "version": "pro",
            "fixed_subset_cases": 25,
            "task_ids": list(SWE_BENCH_PRO_TASK_IDS),
            "subset_id": "horizon-swe-pro-aux-v1",
        },
    }


def build_default_manifest(*, commit: str | None = None) -> BenchmarkManifest:
    """Build the frozen three-model plan without touching provider state."""

    manifest = BenchmarkManifest(commit=commit)
    validate_manifest(manifest_to_dict(manifest))
    return manifest


def manifest_to_dict(manifest: BenchmarkManifest) -> dict[str, Any]:
    """Return a JSON-safe manifest payload with tuples represented as lists."""

    payload = asdict(manifest)
    payload["models"] = [asdict(model) for model in manifest.models]
    payload["agent"]["capabilities"] = list(manifest.agent.capabilities)
    payload["failure_types"] = list(manifest.failure_types)
    payload["task_sets"] = {
        str(name): dict(values) for name, values in (manifest.task_sets or {}).items()
    }
    payload["budget"] = dict(manifest.budget or {})
    return payload


def build_run_manifest(
    *,
    run_id: str,
    commit: str,
    dirty: bool,
    benchmark: Mapping[str, Any],
    task_selection: Mapping[str, Any],
    model: Mapping[str, Any],
    agent: Mapping[str, Any],
    budget: Mapping[str, Any],
    runtime: Mapping[str, Any],
    artifact_root: str,
    run_at: str | None = None,
) -> dict[str, Any]:
    """Create the immutable launch record shared by every benchmark adapter.

    This function is pure: operational launch code resolves Git state, provider
    configuration, task selectors, and the intended artifact root before it
    calls this interface.  The result contains inputs only; scores and failure
    analysis belong to result artifacts produced after the run.
    """

    if not run_id.strip():
        raise ValueError("run manifest requires a run id")
    if not commit.strip():
        raise ValueError("run manifest requires a Git commit")
    if not str(benchmark.get("name", "")).strip():
        raise ValueError("run manifest requires a benchmark name")
    if not str(model.get("requested_id", "")).strip():
        raise ValueError("run manifest requires a requested model id")
    if not artifact_root.strip():
        raise ValueError("run manifest requires an artifact root")
    selection = dict(task_selection)
    selection_json = json.dumps(
        selection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "schema": "horizon-harness.run-manifest.v1",
        "run_id": run_id,
        "source": {"git_commit": commit, "git_dirty": dirty},
        "benchmark": dict(benchmark),
        "task_selection": {
            **selection,
            "sha256": hashlib.sha256(selection_json.encode("utf-8")).hexdigest(),
        },
        "model": dict(model),
        "agent": dict(agent),
        "budget": dict(budget),
        "runtime": dict(runtime),
        "artifacts": {"root": artifact_root},
        "run_at": run_at or datetime.now(timezone.utc).isoformat(),
    }


@dataclass(frozen=True)
class BenchmarkReport:
    """Comparable result metrics for one model and one frozen task set."""

    schema: str = "horizon-harness.benchmark-report.v1"
    benchmark: str = ""
    benchmark_version: str = ""
    model: str = ""
    harness_commit: str = ""
    metrics: Mapping[str, float | int | None] | None = None
    failure_counts: Mapping[str, int] | None = None
    public_reference: bool = False

    def __post_init__(self) -> None:
        metrics = dict(self.metrics or {})
        missing = [name for name in REPORT_METRICS if name not in metrics]
        if missing:
            raise ValueError("benchmark report missing metrics: " + ", ".join(missing))
        unknown_failures = set(self.failure_counts or {}) - set(FAILURE_TYPES)
        if unknown_failures:
            raise ValueError("unknown failure type: " + sorted(unknown_failures)[0])
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "failure_counts", dict(self.failure_counts or {}))


def report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    return asdict(report)


def validate_report(payload: Mapping[str, Any]) -> None:
    """Validate the metric and failure taxonomy contract before publication."""

    if payload.get("schema") != "horizon-harness.benchmark-report.v1":
        raise ValueError("unsupported benchmark report schema")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("benchmark report metrics are required")
    missing = [name for name in REPORT_METRICS if name not in metrics]
    if missing:
        raise ValueError("benchmark report missing metrics: " + ", ".join(missing))
    failures = payload.get("failure_counts") or {}
    if not isinstance(failures, Mapping) or set(failures) - set(FAILURE_TYPES):
        raise ValueError("benchmark report has an unsupported failure type")


def compare_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    metric: str = "score",
) -> dict[str, Any]:
    """Compare two reports only when their frozen identity is compatible.

    Reports from another benchmark version, model, or Harness commit are
    useful context but cannot support an improvement claim. They are returned
    as ``public_reference`` with no delta instead of being silently compared.
    """

    validate_report(baseline)
    validate_report(candidate)
    if metric not in REPORT_METRICS:
        raise ValueError(f"unsupported comparison metric {metric!r}")
    identity = ("benchmark", "benchmark_version", "model", "harness_commit")
    comparable = all(baseline.get(key) == candidate.get(key) for key in identity)
    if not comparable:
        return {
            "metric": metric,
            "comparable": False,
            "public_reference": True,
            "baseline": baseline["metrics"].get(metric),
            "candidate": candidate["metrics"].get(metric),
            "delta": None,
        }
    before = baseline["metrics"].get(metric)
    after = candidate["metrics"].get(metric)
    if before is None or after is None:
        raise ValueError(f"comparison metric {metric!r} must be numeric")
    return {
        "metric": metric,
        "comparable": True,
        "public_reference": False,
        "baseline": before,
        "candidate": after,
        # Keep the serialized comparison stable for decimal report values.
        "delta": round(after - before, 12),
    }


def build_benchmark_report(
    *,
    benchmark: str,
    benchmark_version: str,
    model: ModelBenchmarkConfig,
    harness_commit: str,
    metrics: Mapping[str, float | int | None],
    failure_counts: Mapping[str, int] | None = None,
    public_reference: bool = False,
) -> dict[str, Any]:
    """Build a report while preventing incomparable partial metric records."""

    report = BenchmarkReport(
        benchmark=benchmark,
        benchmark_version=benchmark_version,
        model=model.model,
        harness_commit=harness_commit,
        metrics=metrics,
        failure_counts=failure_counts,
        public_reference=public_reference,
    )
    payload = report_to_dict(report)
    validate_report(payload)
    return payload


def validate_offline_paths(
    *,
    fake_runner: Callable[[], Mapping[str, Any]],
    oracle_runner: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run caller-supplied Fake and Oracle paths before paid evaluation.

    The runners are injected so this module stays independent of Docker and
    provider SDKs. Both paths must return a result mapping with a trace and an
    artifact; the oracle must additionally report a successful reference score.
    """

    fake = dict(fake_runner())
    oracle = dict(oracle_runner())
    for label, result in (("fake", fake), ("oracle", oracle)):
        if not result.get("trace_complete", False):
            raise ValueError(f"{label} offline path did not produce a complete trace")
        if not result.get("artifact_complete", False):
            raise ValueError(
                f"{label} offline path did not produce a complete artifact"
            )
    if not oracle.get("oracle_score_ok", False):
        raise ValueError("oracle offline path did not reproduce the expected score")
    return {"fake": fake, "oracle": oracle, "ready_for_paid_run": True}


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write a JSON manifest atomically enough for local benchmark setup."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an offline HorizonHarness benchmark manifest"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commit", default="unfrozen")
    args = parser.parse_args(argv)
    manifest = build_default_manifest(commit=args.commit)
    payload = manifest_to_dict(manifest)
    if args.output:
        write_manifest(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def validate_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the public manifest contract before a benchmark is launched."""

    if payload.get("schema") != "horizon-harness.benchmark.v1":
        raise ValueError("unsupported benchmark manifest schema")
    models = payload.get("models")
    expected = [asdict(model) for model in MODEL_MATRIX]
    if models != expected:
        raise ValueError("model matrix must match the fixed three-model configuration")
    agent = payload.get("agent")
    if not isinstance(agent, Mapping):
        raise ValueError("manifest agent configuration is required")
    if agent.get("workflow") != "single":
        raise ValueError("benchmark agent workflow must remain single")
    if agent.get("capabilities") != ["bash", "read", "task"]:
        raise ValueError("benchmark capabilities must be bash, read, task")
    for key in ("memory", "skills", "mcp"):
        if agent.get(key) is not False:
            raise ValueError(f"benchmark {key} must be disabled")
    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("benchmark stages are required")
    required = {
        "api_preflight": 5,
        "runtime_reliability": 18,
        "terminal_bench": 20,
        "swe_bench_pro": 25,
    }
    for name, count in required.items():
        stage = stages.get(name)
        if not isinstance(stage, Mapping):
            raise ValueError(f"missing benchmark stage {name!r}")
        actual = stage.get("pilot_cases", stage.get("cases"))
        if actual != count:
            raise ValueError(f"{name} must define {count} cases")
    if tuple(stages["api_preflight"].get("case_types", ())) != API_PREFLIGHT_CASES:
        raise ValueError("API preflight case types drifted")
    if tuple(stages["runtime_reliability"].get("case_ids", ())) != RELIABILITY_CASES:
        raise ValueError("runtime reliability case list drifted")
    if (
        tuple(stages["terminal_bench"].get("pilot_task_ids", ()))
        != TERMINAL_BENCH_PILOT_TASK_IDS
    ):
        raise ValueError("Terminal-Bench pilot task list drifted")
    if tuple(stages["swe_bench_pro"].get("task_ids", ())) != SWE_BENCH_PRO_TASK_IDS:
        raise ValueError("SWE-bench Pro task list drifted")
    if len(set(stages["runtime_reliability"].get("case_ids", ()))) != 18:
        raise ValueError("runtime reliability case list must be unique")
    if len(set(stages["terminal_bench"].get("pilot_task_ids", ()))) != 20:
        raise ValueError("Terminal-Bench pilot task list must be unique")
    if len(set(stages["swe_bench_pro"].get("task_ids", ()))) != 25:
        raise ValueError("SWE-bench Pro task list must be unique")
    if stages["terminal_bench"].get("max_turns") != 150:
        raise ValueError("Terminal-Bench max_turns must be 150")
    if stages["swe_bench_pro"].get("max_turns") != 200:
        raise ValueError("SWE-bench Pro max_turns must be 200")
    if tuple(payload.get("failure_types", ())) != FAILURE_TYPES:
        raise ValueError("failure taxonomy must match the fixed HorizonHarness set")
    task_sets = payload.get("task_sets")
    if not isinstance(task_sets, Mapping):
        raise ValueError("manifest task sets are required")
    budget = payload.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("manifest budget is required")


__all__ = [
    "FAILURE_TYPES",
    "API_PREFLIGHT_CASES",
    "RELIABILITY_CASES",
    "TERMINAL_BENCH_PILOT_TASK_IDS",
    "SWE_BENCH_PRO_TASK_IDS",
    "REPORT_METRICS",
    "MODEL_MATRIX",
    "BenchmarkAgentConfig",
    "BenchmarkManifest",
    "ModelBenchmarkConfig",
    "build_default_manifest",
    "manifest_to_dict",
    "validate_manifest",
    "build_run_manifest",
    "BenchmarkReport",
    "report_to_dict",
    "validate_report",
    "compare_reports",
    "build_benchmark_report",
    "validate_offline_paths",
    "write_manifest",
]


if __name__ == "__main__":  # pragma: no cover - CLI smoke is shell-tested
    raise SystemExit(_cli())
