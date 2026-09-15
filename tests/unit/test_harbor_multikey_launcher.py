"""Tests for model/key selection in the three-shard Harbor launcher."""

from __future__ import annotations

import unittest

from runs.harbor import run_terminal_bench_2_1_baseline_multikey as multikey


class HarborMultikeyLauncherTests(unittest.TestCase):
    def test_flash_model_uses_three_explicit_additional_key_slots(self) -> None:
        args = multikey._build_parser().parse_args(
            [
                "--model",
                "deepseek-ai/deepseek-v4-flash-0731",
                "--key-env",
                "OPENAI_AUTH_TOKEN_4",
                "--key-env",
                "OPENAI_AUTH_TOKEN_5",
                "--key-env",
                "OPENAI_AUTH_TOKEN_6",
                "--run-name",
                "flash-test",
                "--dry-run",
            ]
        )

        launches = multikey._launches(args)
        payload = multikey._safe_launch_payload("flash-test", args.model, launches)

        self.assertEqual(
            [launch.key_env for launch in launches],
            [
                "OPENAI_AUTH_TOKEN_4",
                "OPENAI_AUTH_TOKEN_5",
                "OPENAI_AUTH_TOKEN_6",
            ],
        )
        self.assertTrue(all(len(launch.tasks) == 6 for launch in launches))
        self.assertTrue(
            all(
                "openai/deepseek-ai/deepseek-v4-flash-0731" in launch.command
                for launch in launches
            )
        )
        self.assertEqual(payload["model"], "deepseek-ai/deepseek-v4-flash-0731")
        self.assertEqual(payload["total_concurrency"], 6)

    def test_launcher_rejects_any_key_slot_count_other_than_three(self) -> None:
        args = multikey._build_parser().parse_args(
            ["--key-env", "OPENAI_AUTH_TOKEN_4", "--dry-run"]
        )

        with self.assertRaisesRegex(ValueError, "exactly three"):
            multikey._launches(args)


if __name__ == "__main__":
    unittest.main()
