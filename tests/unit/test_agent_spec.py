"""Contracts for orthogonal agent assembly data."""

from __future__ import annotations

import unittest

from horizon_harness import AgentSpec
from horizon_harness.spec import agent_spec_from_env


class AgentSpecTests(unittest.TestCase):
    def test_capabilities_and_workflow_are_independent_axes(self) -> None:
        capabilities = ("bash", "read", "task")

        specs = [
            AgentSpec(capabilities=capabilities, workflow=workflow)
            for workflow in (
                "single",
                "chain",
                "planner_executor",
                "reflection",
                "routing",
                "parallel",
                "goal_loop",
                "pdr",
            )
        ]

        self.assertTrue(all(spec.capabilities == capabilities for spec in specs))
        self.assertEqual(len({spec.workflow for spec in specs}), 8)
        self.assertFalse(hasattr(specs[0], "flavor"))

    def test_unknown_capability_or_workflow_fails_at_the_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "capability"):
            AgentSpec(capabilities=("bash", "browser"))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "workflow"):
            AgentSpec(workflow="magic")  # type: ignore[arg-type]

    def test_environment_selects_each_axis_without_a_combination_key(self) -> None:
        base = AgentSpec(name="benchmark", role="solver", capabilities=("bash",))

        resolved = agent_spec_from_env(
            base,
            {
                "HORIZON_CAPABILITIES": "bash,read,task",
                "HORIZON_WORKFLOW": "pdr",
            },
        )

        self.assertEqual(resolved.name, "benchmark")
        self.assertEqual(resolved.role, "solver")
        self.assertEqual(resolved.capabilities, ("bash", "read", "task"))
        self.assertEqual(resolved.workflow, "pdr")

    def test_legacy_flavor_selector_fails_with_migration_hint(self) -> None:
        with self.assertRaisesRegex(ValueError, "HORIZON_CAPABILITIES"):
            agent_spec_from_env(AgentSpec(), {"AGENT_FLAVOR": "bash_task"})


if __name__ == "__main__":
    unittest.main()
