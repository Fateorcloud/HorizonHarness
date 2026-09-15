"""Contracts for the single high-level ``AgentSession`` assembly seam."""

from __future__ import annotations

import logging
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable, Sequence

from horizon_harness import AgentSession, AgentSpec, Provider
from horizon_harness.agents.starter import (
    BASH_AGENT_SYSTEM_PROMPT,
    BASH_TASK_ADDENDUM,
    MCP_ADDENDUM,
    SKILLS_ADDENDUM,
    SkillConfig,
    compose_agent_system_prompt,
)
from horizon_harness.agents.toolsets import MCPToolset, Toolset
from horizon_harness.messages import TextBlock
from horizon_harness.skills import SkillRoot
from horizon_harness.tools import AgentTool, ToolResult, text_result


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SKILLS = ROOT / "tests" / "fixtures" / "skills"
FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")

try:
    from mcp import ClientSession
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import (
        create_connected_server_and_client_session as connect_session,
    )

    from horizon_harness.mcp import MCPConnection, MCPServerConfig

    HAS_MCP = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_MCP = False

_SKIP_REASON = "mcp extra not installed (install with: uv sync --extra mcp)"
logging.getLogger("mcp").setLevel(logging.WARNING)


def _static_tool(name: str) -> AgentTool:
    def execute(call_id, args, abort, on_update) -> ToolResult:  # noqa: ANN001
        del call_id, args, abort, on_update
        return text_result(f"{name} ran")

    return AgentTool(
        name=name,
        description=f"test tool {name}",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


class FakeToolset:
    def __init__(self, tools: Sequence[AgentTool]) -> None:
        self._tools = tuple(tools)
        self.opened = False
        self.closed = False

    def __enter__(self) -> "FakeToolset":
        self.opened = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def tools(self) -> Sequence[AgentTool]:
        if not self.opened:
            raise RuntimeError("FakeToolset.tools() before open")
        return self._tools


class ToolsetProtocolTest(unittest.TestCase):
    def test_fake_toolset_satisfies_protocol(self) -> None:
        self.assertIsInstance(FakeToolset([_static_tool("a")]), Toolset)

    def test_session_owns_toolset_lifetime(self) -> None:
        toolset = FakeToolset([_static_tool("extra")])
        with AgentSession(
            FAKE_PROVIDER,
            spec=AgentSpec(capabilities=()),
            tools=(_static_tool("static"),),
            toolsets=(toolset,),
        ) as session:
            self.assertTrue(toolset.opened)
            self.assertEqual(
                [tool.name for tool in session.agent.tools],
                ["static", "extra"],
            )
        self.assertTrue(toolset.closed)

    def test_session_closes_toolset_when_body_raises(self) -> None:
        toolset = FakeToolset([_static_tool("extra")])
        with self.assertRaisesRegex(ValueError, "boom"):
            with AgentSession(
                FAKE_PROVIDER,
                spec=AgentSpec(capabilities=()),
                toolsets=(toolset,),
            ):
                raise ValueError("boom")
        self.assertTrue(toolset.closed)


class ComposePromptTest(unittest.TestCase):
    def test_composes_enabled_fragments_in_order(self) -> None:
        prompt = compose_agent_system_prompt(
            bash=True,
            general_purpose=True,
            skills=True,
            mcp=True,
        )
        self.assertEqual(
            prompt,
            "\n\n".join(
                (
                    BASH_AGENT_SYSTEM_PROMPT,
                    BASH_TASK_ADDENDUM,
                    SKILLS_ADDENDUM,
                    MCP_ADDENDUM,
                )
            ),
        )


class AgentSessionTest(unittest.TestCase):
    def test_agent_is_available_only_while_entered(self) -> None:
        session = AgentSession(FAKE_PROVIDER, spec=AgentSpec(capabilities=()))
        with self.assertRaisesRegex(RuntimeError, "only available"):
            _ = session.agent
        with session:
            self.assertEqual(session.agent.name, "agent")
        with self.assertRaisesRegex(RuntimeError, "only available"):
            _ = session.agent

    def test_capabilities_are_assembled_in_spec_order(self) -> None:
        spec = AgentSpec(
            name="solver",
            capabilities=("bash", "read", "edit", "task"),
        )
        with AgentSession(FAKE_PROVIDER, spec=spec, cwd=ROOT) as session:
            self.assertEqual(
                [tool.name for tool in session.agent.tools],
                ["bash", "read", "edit", "task"],
            )

    def test_skills_capability_installs_state_initializer(self) -> None:
        skills = SkillConfig(
            roots=(SkillRoot(str(FIXTURE_SKILLS), "repo"),),
            cwd=str(FIXTURE_SKILLS),
        )
        spec = AgentSpec(capabilities=("bash", "read", "skills"))
        with AgentSession(
            FAKE_PROVIDER,
            spec=spec,
            cwd=FIXTURE_SKILLS,
            skills=skills,
        ) as session:
            state, events = session.run("echo hello", max_turns=3)
            tuple(events)
        menu = next(message for message in state.messages if message.sender == "skills")
        self.assertIn("echo-fixture", menu.content[0].text)

    def test_skills_config_requires_the_capability(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires 'skills'"):
            AgentSession(
                FAKE_PROVIDER,
                spec=AgentSpec(capabilities=("bash",)),
                skills=SkillConfig(),
            )


def _demo_server() -> "FastMCP":
    server = FastMCP("demo")

    @server.tool(description="Echo text back as plain text.")
    def echo(text: str) -> str:
        return f"echo: {text}"

    return server


def _in_memory_connect() -> Callable[["MCPServerConfig"], "MCPConnection"]:
    server = _demo_server()

    def connect(config: "MCPServerConfig") -> "MCPConnection":
        @asynccontextmanager
        async def factory() -> AsyncIterator["ClientSession"]:
            async with connect_session(server._mcp_server) as session:
                yield session

        return MCPConnection(factory, name=config.name).open()

    return connect


@unittest.skipUnless(HAS_MCP, _SKIP_REASON)
class MCPToolsetTest(unittest.TestCase):
    def test_toolset_requires_context(self) -> None:
        toolset = MCPToolset(MCPServerConfig.stdio("demo", "noop"))
        with self.assertRaises(RuntimeError):
            toolset.tools()

    def test_session_exposes_prefixed_mcp_tool(self) -> None:
        config = MCPServerConfig.stdio("demo", "noop")
        with AgentSession(
            FAKE_PROVIDER,
            spec=AgentSpec(capabilities=()),
            mcp_servers=(config,),
            connect=_in_memory_connect(),
        ) as session:
            echo = next(
                tool for tool in session.agent.tools if tool.name == "demo_echo"
            )
            result = echo.execute("c1", {"text": "hi"}, lambda: False, None)
            self.assertEqual(result.content, (TextBlock("echo: hi"),))


if __name__ == "__main__":
    unittest.main()
