"""Public assembly contract for AgentSession."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from horizon_harness import AgentSession, AgentSpec, Provider


FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


class AgentSessionAssemblyTests(unittest.TestCase):
    def test_spec_assembles_capabilities_without_a_flavor_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = AgentSpec(
                name="solver",
                capabilities=("bash", "read", "edit", "task"),
                workflow="single",
            )

            with AgentSession(FAKE_PROVIDER, spec=spec, cwd=Path(tmp)) as session:
                self.assertEqual(session.agent.name, "solver")
                self.assertEqual(
                    [tool.name for tool in session.agent.tools],
                    ["bash", "read", "edit", "task"],
                )

    def test_workflow_uses_a_recipe_over_the_same_agent_loop(self) -> None:
        spec = AgentSpec(capabilities=(), workflow="chain")

        with AgentSession(FAKE_PROVIDER, spec=spec, max_turns=2) as session:
            state, events = session.run("Return a short answer.")
            tuple(events)

        self.assertEqual(state.data["recipe"]["workflow"], "chain")
        self.assertEqual(len(state.data["recipe"]["steps"]), 2)
        self.assertTrue(any(message.kind == "final" for message in state.messages))


if __name__ == "__main__":
    unittest.main()
