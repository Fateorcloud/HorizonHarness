"""Contract tests for the ProgramBench suite after the HorizonHarness refactor."""

from __future__ import annotations

import base64
import io
import tempfile
import unittest
import tarfile
from pathlib import Path
from unittest import mock

from horizon_harness.evals.suites.programbench import container
from evals.programbench import harness
from evals.programbench.evaluate_submissions import default_eval_dir
from evals.programbench.suite import ProgrambenchSuite


def _instance(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "instance_id": "testorg__calculator.abc1234",
        "image_name": "programbench/testorg_1776_calculator.abc1234",
        "language": "bash",
        "repository": "testorg/calculator",
        "commit": "abc1234567890",
        "branches": {"main": {}},
        "tests": {"main": [{"name": "t"}]},
    }
    record.update(overrides)
    return record


class ProgrambenchSuiteTest(unittest.TestCase):
    def test_launch_spec_is_data_and_uses_command_isolation(self) -> None:
        spec = ProgrambenchSuite().launch_spec(_instance())
        self.assertEqual(
            spec.image, "programbench/testorg_1776_calculator.abc1234:task_cleanroom"
        )
        self.assertEqual(spec.workdir, "/workspace")
        self.assertEqual(spec.network_mode, "host")
        self.assertEqual(spec.cap_add, ("SYS_ADMIN",))

    def test_task_input_drops_identity_and_gold(self) -> None:
        view = ProgrambenchSuite().task_input(_instance())
        self.assertEqual(view["instance_id"], "testorg__calculator.abc1234")
        self.assertIn("language", view)
        for field in ("repository", "commit", "image_name", "branches", "tests"):
            self.assertNotIn(field, view)

    def test_programbench_scores_outside_the_container(self) -> None:
        self.assertIsNone(ProgrambenchSuite().eval_inputs(_instance()))
        self.assertEqual(
            ProgrambenchSuite(cap_add=()).launch_spec(_instance()).cap_add, ()
        )


class ProgrambenchContainerTest(unittest.TestCase):
    def test_agent_identity_uses_capability_and_workflow_axes(self) -> None:
        spec = container.agent_spec()
        self.assertEqual(spec.capabilities, ("bash", "read", "task"))
        self.assertEqual(spec.workflow, "single")
        self.assertNotIn("flavor", repr(spec).lower())

    def test_build_task_mentions_observation_rules_and_submit(self) -> None:
        task = container.build_task({}, workdir="/workspace")
        self.assertIn("reverse-engineering benchmark", task)
        self.assertIn("provided `./executable`", task)
        self.assertIn("submit", task.lower())

    def test_submit_tool_requires_summary_and_terminates(self) -> None:
        tool = container.make_submit_tool()
        missing = tool.execute("call-1", {}, lambda: False, None)
        self.assertTrue(missing.is_error)
        accepted = tool.execute("call-2", {"summary": "checked"}, lambda: False, None)
        self.assertTrue(accepted.terminate)

    def test_session_options_are_suite_extensions(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                container, "_detect_network_isolation", return_value=True
            ),
        ):
            options = container.agent_session_options(
                provider=container.Provider(id="fake", api="fake", model="fake-model"),
                cwd=Path(tmp),
                context={"runtime": {"max_turns": 3}},
            )
        self.assertIn("tools", options)
        self.assertIn("hooks", options)
        self.assertEqual(options["bash_exec_prefix"], container.NET_ISOLATION_PREFIX)

    def test_extract_result_packages_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "compile.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            result = container.extract_result(workspace, {})
        payload = base64.b64decode(result["submission_tar_b64"])
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            self.assertIn("./compile.sh", archive.getnames())


class ProgrambenchHarnessTest(unittest.TestCase):
    def test_default_eval_dir_isolated_by_selected_instances(self) -> None:
        root = Path("evals/out/programbench")
        self.assertEqual(default_eval_dir(root, "run-1", None), root / "run-1_eval")
        self.assertEqual(
            default_eval_dir(root, "run-1", ["case-1"]), root / "run-1_eval_case-1"
        )

    def test_container_environment_uses_horizon_namespace(self) -> None:
        env = {
            "OPENAI_MODEL": "m",
            "OPENAI_AUTH_TOKEN": "t",
            "HORIZON_WORKFLOW": "single",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            passed = harness.container_environment("openai")
        self.assertEqual(passed["HORIZON_WORKFLOW"], "single")
        self.assertNotIn("AGENT_FLAVOR", passed)


if __name__ == "__main__":
    unittest.main()
