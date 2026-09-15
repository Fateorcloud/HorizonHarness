"""Contract tests for the SWE-bench container seam.

These tests intentionally exercise the post-Flavor API: suite identity is an
``AgentSpec`` and assembly is delegated to the generic ``AgentSession`` path.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from horizon_harness import AgentSpec
from horizon_harness.evals.suites.swebench import container


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True)

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )

    run("init", "-q")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "Test")
    (root / "a.txt").write_text("original\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "baseline")
    return root


class SwebenchAgentSpecTest(unittest.TestCase):
    def test_identity_uses_independent_default_axes(self) -> None:
        spec = container.agent_spec()
        self.assertIsInstance(spec, AgentSpec)
        self.assertEqual(spec.name, container.AGENT_NAME)
        self.assertEqual(spec.capabilities, ("bash", "read", "task"))
        self.assertEqual(spec.workflow, "single")

    def test_spec_does_not_encode_a_flavor_string(self) -> None:
        spec = container.agent_spec()
        self.assertFalse(hasattr(spec, "flavor"))
        self.assertNotIn("flavor", repr(spec).lower())


class SwebenchTaskTest(unittest.TestCase):
    def test_build_task_contains_problem_and_boundaries(self) -> None:
        task = container.build_task(
            {
                "problem_statement": "Fix the parser.",
                "requirements": "Keep the public API.",
                "interface": "parse(text)",
            },
            workdir="/workspace",
        )
        self.assertIn("Fix the parser.", task)
        self.assertIn("Keep the public API.", task)
        self.assertIn("parse(text)", task)
        self.assertIn("DO NOT MODIFY: Tests", task)

    def test_optional_problem_fields_are_empty_safe(self) -> None:
        task = container.build_task({}, workdir="/workspace")
        self.assertIn("<pr_description>", task)
        self.assertIn("</instructions>", task)


class SwebenchPatchContractTest(unittest.TestCase):
    def test_prepare_and_extract_collect_workspace_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _init_repo(Path(tmp) / "repo")
            context = container.prepare(workspace, {"instance_id": "case-1"})
            (workspace / "a.txt").write_text("changed\n", encoding="utf-8")
            result = container.extract_result(
                workspace, {"instance_id": "case-1"}, context=context
            )
        self.assertIn("a.txt", result["model_patch"])
        self.assertIn("changed", result["model_patch"])

    def test_oracle_applies_gold_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _init_repo(Path(tmp) / "repo")
            patch = """diff --git a/a.txt b/a.txt
index 5b6f2a3..e69de29 100644
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-original
+oracle
"""
            container.apply_oracle(workspace, {"patch": patch})
            self.assertEqual((workspace / "a.txt").read_text(), "oracle\n")

    def test_oracle_rejects_missing_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _init_repo(Path(tmp) / "repo")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                container.apply_oracle(workspace, {})

    def test_memory_artifacts_match_scored_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = _init_repo(Path(tmp) / "repo")
            context = container.prepare(workspace, {"instance_id": "case-1"})
            (workspace / "a.txt").write_text("changed\n", encoding="utf-8")
            artifacts = container.memory_artifacts(
                workspace, {"instance_id": "case-1"}, context=context
            )
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].name, "model_patch.diff")
        self.assertIn("changed", artifacts[0].content)


class SwebenchChainContextTest(unittest.TestCase):
    def test_chain_policy_can_disable_compression_without_flavor(self) -> None:
        policy = container.chain_context_policy(
            provider=container.Provider(id="fake", api="fake", model="fake-model"),
            request_extra=None,
            config={"config": {"compression_strategy": "none"}},
        )
        self.assertIsNotNone(policy)


if __name__ == "__main__":
    unittest.main()
