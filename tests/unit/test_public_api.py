"""Public package contract for HorizonHarness."""

from __future__ import annotations

import unittest


class PublicApiTests(unittest.TestCase):
    def test_root_package_exports_only_the_runtime_contract(self) -> None:
        import horizon_harness

        self.assertEqual(
            set(horizon_harness.__all__),
            {
                "Agent",
                "State",
                "Message",
                "Event",
                "AgentTool",
                "ToolResult",
                "AgentSpec",
                "AgentSession",
                "Provider",
                "run",
            },
        )


if __name__ == "__main__":
    unittest.main()
