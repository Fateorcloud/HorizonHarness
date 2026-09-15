"""Focused tests for the bash + task delegation agent preset.

The fake LLM adapter is bash-aware but not task-aware (see
``horizon_harness.llm.adapters.fake``), so these tests drive the
parent's ``generate`` callable directly with a small stub. That keeps
the focus on the wiring this preset introduces:

* parent agent exposes both ``bash`` and ``task`` tools,
* parent's system prompt is the bash-agent prompt plus a small addendum,
* task dispatch picks ``subagent_type="general-purpose"`` and the worker's
  bash final message rides back as the parent's tool result,
* both parent and worker share the requested ``cwd``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from horizon_harness import AgentSession, AgentSpec
from horizon_harness.messages import (
    Message,
    ToolCallBlock,
    assistant_message,
    message_text,
    tool_results_of,
)
from horizon_harness.agents.starter import (
    BASH_AGENT_SYSTEM_PROMPT,
    BASH_TASK_AGENT_SYSTEM_PROMPT,
    BASH_TASK_ADDENDUM,
    GENERAL_PURPOSE_AGENT_DEFAULT_NAME,
)
from horizon_harness.llm import Provider


ROOT = Path(__file__).resolve().parents[2]
FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


class BashTaskAgentTest(unittest.TestCase):
    def test_parent_exposes_bash_and_task_tools(self) -> None:
        with AgentSession(
            FAKE_PROVIDER,
            spec=AgentSpec(name="bash_task_agent", capabilities=("bash", "task")),
            cwd=ROOT,
        ) as session:
            tool_names = sorted(tool.name for tool in session.agent.tools)
        self.assertEqual(tool_names, ["bash", "task"])

    def test_task_tool_lists_only_the_general_purpose_subagent(self) -> None:
        with AgentSession(
            FAKE_PROVIDER,
            spec=AgentSpec(name="bash_task_agent", capabilities=("bash", "task")),
            cwd=ROOT,
        ) as session:
            task_tool_def = next(
                tool for tool in session.agent.tools if tool.name == "task"
            )
        self.assertEqual(
            list(task_tool_def.parameters["properties"]["subagent_type"]["enum"]),
            [GENERAL_PURPOSE_AGENT_DEFAULT_NAME],
        )
        self.assertIn(GENERAL_PURPOSE_AGENT_DEFAULT_NAME, task_tool_def.description)

    def test_system_prompt_extends_bash_prompt_with_short_addendum(self) -> None:
        self.assertTrue(
            BASH_TASK_AGENT_SYSTEM_PROMPT.startswith(BASH_AGENT_SYSTEM_PROMPT),
            "Composed prompt should reuse the bash-agent prompt as its base.",
        )
        self.assertIn(BASH_TASK_ADDENDUM, BASH_TASK_AGENT_SYSTEM_PROMPT)
        addendum_lines = BASH_TASK_ADDENDUM.count(". ") + 1
        self.assertLessEqual(
            addendum_lines,
            5,
            f"Addendum should stay short; got ~{addendum_lines} sentences.",
        )

    def test_delegating_to_general_purpose_returns_sub_agent_final_as_tool_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "marker.txt").write_text("worker-saw-this\n")

            with AgentSession(
                FAKE_PROVIDER,
                spec=AgentSpec(name="bash_task_agent", capabilities=("bash", "task")),
                cwd=workspace,
            ) as session:
                session.agent.generate = _make_parent_generate(
                    delegated_task="Use bash to run command: `cat marker.txt`",
                )
                state, events = session.run("delegate the read", max_turns=4)
                tuple(events)

        tool_result_msg = next(
            message
            for message in reversed(state.messages)
            if message.kind == "tool_result"
        )
        blocks = tool_results_of(tool_result_msg.content)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tool_name, "task")
        self.assertIn("worker-saw-this", message_text(tool_result_msg))

        final = next(
            message for message in reversed(state.messages) if message.kind == "final"
        )
        self.assertEqual(final.sender, "bash_task_agent")
        self.assertIn("done", message_text(final))


def _make_parent_generate(*, delegated_task: str):
    """Build a tiny deterministic parent generate function.

    Turn 1: emit a single ``task`` tool call asking the general-purpose worker
    to run a small bash command. Turn 2 (after the tool result comes back):
    emit a one-line final.
    """

    call_count = 0

    def generate(visible: list[Message]) -> Message:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return assistant_message(
                (
                    ToolCallBlock(
                        id="task_1",
                        name="task",
                        arguments={
                            "subagent_type": GENERAL_PURPOSE_AGENT_DEFAULT_NAME,
                            "task": delegated_task,
                        },
                    ),
                ),
                sender="bash_task_agent",
                target="user",
                kind="step",
            )
        return assistant_message(
            "delegation done",
            sender="bash_task_agent",
            target="user",
            kind="final",
        )

    return generate
