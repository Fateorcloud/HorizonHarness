"""Tests for deterministic multi-key Harbor Baseline runs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from runs.lib.harbor_shards import (
    build_baseline_shard_command,
    merge_harbor_results,
    partition_tasks,
    read_task_names,
)


class HarborShardTests(unittest.TestCase):
    def test_fixed_task_file_partitions_into_three_ordered_six_task_shards(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        tasks = read_task_names(root / "runs/harbor/terminal_bench_2_1_fixed_18.txt")

        shards = partition_tasks(tasks, shard_count=3, tasks_per_shard=6)

        self.assertEqual([len(shard) for shard in shards], [6, 6, 6])
        self.assertEqual(tuple(task for shard in shards for task in shard), tasks)
        self.assertNotIn("code-from-image", tasks)
        self.assertNotIn("gcode-to-text", tasks)

    def test_shard_command_contains_tasks_but_never_a_key(self) -> None:
        command = build_baseline_shard_command(
            python_executable="/venv/bin/python",
            dotenv=Path(".env"),
            jobs_dir=Path("evals/out/harbor/jobs"),
            job_name="baseline-run-key1",
            tasks=("task-a", "task-b"),
            model="deepseek-ai/deepseek-v4-flash-0731",
        )

        joined = " ".join(command)
        self.assertEqual(
            command[:4], ["/venv/bin/python", "-m", "runs.run_bench", "harbor"]
        )
        self.assertIn("--harbor-agent terminus-2", joined)
        self.assertIn("--n-concurrent 2", joined)
        self.assertIn("--model openai/deepseek-ai/deepseek-v4-flash-0731", joined)
        self.assertIn("--include-task-name terminal-bench/task-a", joined)
        self.assertNotIn("AUTH_TOKEN", joined)
        self.assertNotIn("API_KEY", joined)

    def test_merge_sums_counts_and_recomputes_reward_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "job-1/result.json"
            second = root / "job-2/result.json"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "started_at": "2026-09-04T00:00:00Z",
                        "updated_at": "2026-09-04T01:00:00Z",
                        "finished_at": "2026-09-04T01:00:00Z",
                        "n_total_trials": 2,
                        "stats": {
                            "n_completed_trials": 2,
                            "n_errored_trials": 0,
                            "n_running_trials": 0,
                            "n_pending_trials": 0,
                            "n_cancelled_trials": 0,
                            "n_retries": 0,
                            "evals": {
                                "terminus-2__terminal-bench/terminal-bench-2-1": {
                                    "n_trials": 2,
                                    "n_errors": 0,
                                    "metrics": [{"mean": 0.5}],
                                    "pass_at_k": {},
                                    "reward_stats": {
                                        "reward": {"1.0": ["a"], "0.0": ["b"]}
                                    },
                                    "exception_stats": {},
                                }
                            },
                            "n_input_tokens": 10,
                            "n_cache_tokens": 2,
                            "n_output_tokens": 4,
                            "cost_usd": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "id": "two",
                        "started_at": "2026-09-04T00:30:00Z",
                        "updated_at": "2026-09-04T02:00:00Z",
                        "finished_at": "2026-09-04T02:00:00Z",
                        "n_total_trials": 1,
                        "stats": {
                            "n_completed_trials": 1,
                            "n_errored_trials": 0,
                            "n_running_trials": 0,
                            "n_pending_trials": 0,
                            "n_cancelled_trials": 0,
                            "n_retries": 1,
                            "evals": {
                                "terminus-2__terminal-bench/terminal-bench-2-1": {
                                    "n_trials": 1,
                                    "n_errors": 0,
                                    "metrics": [{"mean": 1.0}],
                                    "pass_at_k": {},
                                    "reward_stats": {"reward": {"1.0": ["c"]}},
                                    "exception_stats": {"AgentTimeoutError": ["c"]},
                                }
                            },
                            "n_input_tokens": 5,
                            "n_cache_tokens": 1,
                            "n_output_tokens": 3,
                            "cost_usd": 0.2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            merged = merge_harbor_results(
                [first, second], merged_job_name="baseline-merged"
            )

        self.assertEqual(merged["schema"], "horizon-harness.harbor-shards.v1")
        self.assertEqual(merged["n_total_trials"], 3)
        self.assertEqual(merged["stats"]["n_completed_trials"], 3)
        self.assertEqual(merged["stats"]["n_retries"], 1)
        self.assertEqual(merged["stats"]["n_input_tokens"], 15)
        self.assertAlmostEqual(merged["stats"]["cost_usd"], 0.3)
        evaluation = merged["stats"]["evals"][
            "terminus-2__terminal-bench/terminal-bench-2-1"
        ]
        self.assertEqual(evaluation["n_trials"], 3)
        self.assertAlmostEqual(evaluation["metrics"][0]["mean"], 2 / 3)
        self.assertEqual(evaluation["reward_stats"]["reward"]["1.0"], ["a", "c"])
        self.assertEqual(len(merged["source_results"]), 2)


if __name__ == "__main__":
    unittest.main()
