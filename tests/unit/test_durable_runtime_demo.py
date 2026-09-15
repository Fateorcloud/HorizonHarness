import tempfile
import unittest
from pathlib import Path

from runs.demos.durable_runtime_demo import demo_acceptance_contract, run_demo
from horizon_harness.workspace.verification import run_verifier


class DurableRuntimeDemoTests(unittest.TestCase):
    def test_demo_crashes_recovers_verifies_and_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_demo(Path(tmp) / "demo")
        self.assertEqual(result["crash_exit_code"], 91)
        self.assertEqual(result["files"], {"a": "1", "b": "2", "c": "3"})
        self.assertEqual(result["next_model_calls"], 1)
        self.assertEqual(result["bundle_commit_count"], 1)
        self.assertEqual(result["verification_outcome"], "passed")
        self.assertEqual(result["status"], "completed")

    def test_demo_contract_rejects_an_incorrect_recovered_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo"
            run_demo(output)
            (output / "code" / "b.txt").write_text("wrong", encoding="utf-8")

            receipt = run_verifier(
                "negative-demo-check",
                output / "code",
                demo_acceptance_contract(),
            )

        self.assertEqual(receipt.outcome, "code_failed")
