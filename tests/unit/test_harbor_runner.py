"""Tests for the in-container Harbor runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from horizon_harness.evals.harbor import runner


class HarborRunnerTest(unittest.TestCase):
    def test_default_max_turns_is_150(self) -> None:
        args = runner.parse_args(["--instruction", "do the task"])

        self.assertEqual(args.max_turns, 150)

    def test_max_output_tokens_is_an_explicit_optional_budget(self) -> None:
        args = runner.parse_args(
            ["--instruction", "do the task", "--max-output-tokens", "4096"]
        )

        self.assertEqual(args.max_output_tokens, 4096)

    def test_load_instruction_prefers_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instruction.txt"
            path.write_text("solve from file", encoding="utf-8")
            args = runner.parse_args(
                ["--instruction", "inline", "--instruction-file", str(path)]
            )
            self.assertEqual(runner.load_instruction(args), "solve from file")

    def test_fake_provider_writes_summary_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            trace = root / "trajectory.jsonl"

            code = runner.main(
                [
                    "--instruction",
                    "Run command: `pwd`",
                    "--provider",
                    "fake",
                    "--max-turns",
                    "2",
                    "--cwd",
                    str(root),
                    "--summary-path",
                    str(summary),
                    "--trace-path",
                    str(trace),
                ]
            )

            self.assertEqual(code, 0)
            data = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["provider"], "fake")
            self.assertEqual(data["capabilities"], ["bash", "read", "task"])
            self.assertEqual(data["workflow"], "single")
            self.assertTrue(trace.exists())
            self.assertIn(
                "horizon-harness.trajectory",
                trace.read_text(encoding="utf-8"),
            )
