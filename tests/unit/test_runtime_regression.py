import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runs.dev.runtime_regression import main


class RuntimeRegressionTests(unittest.TestCase):
    def test_manifest_runs_in_independent_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixture"
            root.mkdir()
            (root / "x").write_text("1")
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {"cases": [{"id": "a", "root": str(root), "verifier": "test -f x"}]}
                )
            )
            output = Path(tmp) / "out"
            self.assertEqual(
                main(["--manifest", str(manifest), "--output-dir", str(output)]), 0
            )
            rows = json.loads((output / "results.json").read_text())
            self.assertEqual(rows[0]["execution_contract"], "deterministic_fixture")
            self.assertEqual(rows[0]["evidence_kind"], "deterministic_verifier")
            self.assertEqual(rows[0]["status"], "passed")
            original = (output / "results.json").read_bytes()
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                main(["--manifest", str(manifest), "--output-dir", str(output)])
            self.assertEqual((output / "results.json").read_bytes(), original)

    def test_manifest_rejects_duplicate_and_missing_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixture"
            root.mkdir()
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"id": "a", "root": str(root), "verifier": "true"},
                            {"id": "a", "root": str(root), "verifier": "true"},
                        ]
                    }
                )
            )
            with mock.patch("runs.dev.runtime_regression.subprocess.run") as execute:
                with self.assertRaisesRegex(SystemExit, "duplicate"):
                    main(
                        [
                            "--manifest",
                            str(manifest),
                            "--output-dir",
                            str(Path(tmp) / "out"),
                        ]
                    )
                execute.assert_not_called()
            self.assertFalse((Path(tmp) / "out").exists())

    def test_nested_output_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixture"
            root.mkdir()
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {"cases": [{"id": "a", "root": "fixture", "verifier": "true"}]}
                )
            )
            with mock.patch("runs.dev.runtime_regression.subprocess.run") as execute:
                with self.assertRaisesRegex(SystemExit, "separate"):
                    main(
                        ["--manifest", str(manifest), "--output-dir", str(root / "out")]
                    )
                execute.assert_not_called()

    def test_manifest_rejects_debug_fork_ancestry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixture"
            root.mkdir()
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "fork",
                                "root": str(root),
                                "verifier": "true",
                                "debug_fork": True,
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(SystemExit, "debug fork"):
                main(
                    [
                        "--manifest",
                        str(manifest),
                        "--output-dir",
                        str(Path(tmp) / "out"),
                    ]
                )
