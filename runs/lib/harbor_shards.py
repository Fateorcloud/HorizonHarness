"""Pure helpers for deterministic sharded Harbor jobs and result aggregation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


BASELINE_MODEL = "deepseek-ai/deepseek-v4-pro-0813"
BASELINE_DATASET = "terminal-bench/terminal-bench-2-1"


def read_task_names(path: str | Path) -> tuple[str, ...]:
    """Read a reviewable newline-delimited task list."""

    tasks = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"task list contains duplicates: {path}")
    return tasks


def partition_tasks(
    tasks: Sequence[str], *, shard_count: int, tasks_per_shard: int
) -> tuple[tuple[str, ...], ...]:
    """Split an exact task set into stable contiguous shards."""

    expected = shard_count * tasks_per_shard
    if len(tasks) != expected:
        raise ValueError(f"expected {expected} tasks, got {len(tasks)}")
    return tuple(
        tuple(tasks[offset : offset + tasks_per_shard])
        for offset in range(0, expected, tasks_per_shard)
    )


def build_baseline_shard_command(
    *,
    python_executable: str,
    dotenv: Path,
    jobs_dir: Path,
    job_name: str,
    tasks: Sequence[str],
    dataset: str = BASELINE_DATASET,
    model: str = BASELINE_MODEL,
) -> list[str]:
    """Build one six-task Terminus-2 command; credentials stay in its env."""

    harbor_model = model if model.startswith("openai/") else f"openai/{model}"
    command = [
        python_executable,
        "-m",
        "runs.run_bench",
        "harbor",
        "--dotenv",
        str(dotenv),
        "--dataset",
        dataset,
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
        "--harbor-agent",
        "terminus-2",
        "--model",
        harbor_model,
        "--n-concurrent",
        "2",
        "--max-turns",
        "150",
        "--max-output-tokens",
        "4096",
        "--timeout-multiplier",
        "3",
        "--agent-timeout-multiplier",
        "3",
        "--max-retries",
        "1",
        "--debug",
        "--json",
    ]
    for task in tasks:
        qualified = task if "/" in task else f"terminal-bench/{task}"
        command.extend(["--include-task-name", qualified])
    return command


def _sum_optional(values: Iterable[Any]) -> int | float | None:
    items = list(values)
    if not items or any(value is None for value in items):
        return None
    return sum(items)


def _merge_string_lists(target: dict[str, list[str]], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, values in source.items():
        bucket = target.setdefault(str(key), [])
        if isinstance(values, list):
            bucket.extend(str(value) for value in values)


def _merge_evals(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        evaluations = payload.get("stats", {}).get("evals", {})
        if not isinstance(evaluations, dict):
            continue
        for name, evaluation in evaluations.items():
            if not isinstance(evaluation, dict):
                continue
            target = merged.setdefault(
                name,
                {
                    "n_trials": 0,
                    "n_errors": 0,
                    "metrics": [],
                    "pass_at_k": {},
                    "reward_stats": {"reward": {}},
                    "exception_stats": {},
                },
            )
            target["n_trials"] += int(evaluation.get("n_trials", 0))
            target["n_errors"] += int(evaluation.get("n_errors", 0))
            rewards = evaluation.get("reward_stats", {}).get("reward", {})
            _merge_string_lists(target["reward_stats"]["reward"], rewards)
            _merge_string_lists(
                target["exception_stats"], evaluation.get("exception_stats", {})
            )

    for evaluation in merged.values():
        reward_groups = evaluation["reward_stats"]["reward"]
        weighted_total = 0.0
        reward_count = 0
        for reward, trial_ids in reward_groups.items():
            try:
                numeric_reward = float(reward)
            except ValueError:
                continue
            weighted_total += numeric_reward * len(trial_ids)
            reward_count += len(trial_ids)
        mean = weighted_total / reward_count if reward_count else 0.0
        evaluation["metrics"] = [{"mean": mean}]
    return merged


def merge_harbor_results(
    result_paths: Sequence[str | Path], *, merged_job_name: str
) -> dict[str, Any]:
    """Merge disjoint Harbor shard summaries into one auditable aggregate."""

    if not result_paths:
        raise ValueError("at least one Harbor result is required")
    paths = [Path(path).resolve() for path in result_paths]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    stats = [payload.get("stats", {}) for payload in payloads]
    count_fields = (
        "n_completed_trials",
        "n_errored_trials",
        "n_running_trials",
        "n_pending_trials",
        "n_cancelled_trials",
        "n_retries",
    )
    merged_stats: dict[str, Any] = {
        field: sum(int(item.get(field, 0)) for item in stats) for field in count_fields
    }
    merged_stats["evals"] = _merge_evals(payloads)
    for field in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"):
        merged_stats[field] = _sum_optional(item.get(field) for item in stats)

    started = [
        payload.get("started_at") for payload in payloads if payload.get("started_at")
    ]
    updated = [
        payload.get("updated_at") for payload in payloads if payload.get("updated_at")
    ]
    finished_values = [payload.get("finished_at") for payload in payloads]
    finished = (
        max(value for value in finished_values if value)
        if all(finished_values)
        else None
    )
    return {
        "schema": "horizon-harness.harbor-shards.v1",
        "id": merged_job_name,
        "started_at": min(started) if started else None,
        "updated_at": max(updated) if updated else None,
        "finished_at": finished,
        "n_total_trials": sum(
            int(payload.get("n_total_trials", 0)) for payload in payloads
        ),
        "stats": merged_stats,
        "source_results": [str(path) for path in paths],
    }
