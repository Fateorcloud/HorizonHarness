"""The single high-level assembly entry point for HorizonHarness agents.

``AgentSession`` resolves an :class:`~horizon_harness.spec.AgentSpec`, opens
resource-bearing toolsets, assembles capability tools, and runs either the
single agent loop or a recipe over that same loop.  There are intentionally no
combined selectors or parallel factory APIs in this module.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

from horizon_harness.context_view import ContextPolicy
from horizon_harness.core import Agent, ContextBuilderFn, StateInitFn
from horizon_harness.hooks import HookMap
from horizon_harness.llm import Provider as LLMProvider
from horizon_harness.llm import LLMRequest, LLMResponse
from horizon_harness.llm_agent import make_llm_agent
from horizon_harness.messages import ContentInput
from horizon_harness.protocols import Event
from horizon_harness.spec import AgentSpec
from horizon_harness.skills import (
    SkillMetadata,
    SkillRoot,
    init_state_with_skills,
)
from horizon_harness.state import State
from horizon_harness.tools import AbortFlag, AgentTool
from horizon_harness.tools.bash import make_bash_tool
from horizon_harness.tools.edit import make_edit_tool
from horizon_harness.tools.read import make_read_tool
from horizon_harness.tools.task import task_tool

from .toolsets import MCPToolset, Toolset

if TYPE_CHECKING:
    from horizon_harness.mcp import MCPConnection, MCPServerConfig


@dataclass(frozen=True)
class SkillConfig:
    """How a session advertises and injects agent skills.

    Mirrors the keyword surface of
    :func:`~horizon_harness.skills.init_state_with_skills`: ``skills``
    (pre-discovered metadata) takes precedence over ``roots`` (discovery roots);
    ``preload``
    names have their bodies injected up front; ``cwd`` scopes default
    discovery. When ``enabled`` is false the session runs the plain loop even
    though a config is present.
    """

    enabled: bool = True
    roots: Sequence[SkillRoot] | None = None
    skills: Sequence[SkillMetadata] | None = None
    preload: Sequence[str] = ()
    cwd: str = "."


def _skills_init_state(config: SkillConfig) -> StateInitFn:
    """Adapt a `SkillConfig` into a core `Agent.init_state` callable.

    The bridge between the agents layer and the core state initializer hook: it
    binds the config's discovery keywords and forwards to
    :func:`~horizon_harness.skills.init_state_with_skills`, so the resulting
    agent advertises skills on every ``run`` without a session. Skills are
    text-only, so a multimodal task is rejected (the directive parser needs a
    string).
    """

    def init_state(agent: Agent, task: ContentInput) -> State:
        if not isinstance(task, str):
            raise TypeError("skills require a text task, not multimodal content")
        return init_state_with_skills(
            agent,
            task,
            skills=config.skills,
            roots=config.roots,
            preload=config.preload,
            cwd=config.cwd,
        )

    return init_state


def _capability_tools(
    provider: LLMProvider,
    spec: AgentSpec,
    *,
    cwd: str | Path | None,
    extra_tools: Sequence[AgentTool],
    request_extra: Mapping[str, Any] | None,
    context_policy: ContextPolicy | None,
    bash_exec_prefix: tuple[str, ...],
) -> list[AgentTool]:
    """Build the deterministic tool list named by ``AgentSpec.capabilities``."""

    tools: list[AgentTool] = []
    for capability in spec.capabilities:
        if capability == "bash":
            tools.append(make_bash_tool(cwd=cwd, exec_prefix=bash_exec_prefix))
        elif capability == "read":
            tools.append(make_read_tool(cwd=cwd))
        elif capability == "edit":
            tools.append(make_edit_tool(cwd=cwd))
        elif capability == "task":
            worker_tools: list[AgentTool] = [
                make_bash_tool(cwd=cwd, exec_prefix=bash_exec_prefix)
            ]
            if "read" in spec.capabilities:
                worker_tools.append(make_read_tool(cwd=cwd))
            if "edit" in spec.capabilities:
                worker_tools.append(make_edit_tool(cwd=cwd))
            worker = make_llm_agent(
                name=GENERAL_PURPOSE_AGENT_DEFAULT_NAME,
                provider=provider,
                role=GENERAL_PURPOSE_AGENT_DEFAULT_ROLE,
                tools=worker_tools,
                system_prompt=GENERAL_PURPOSE_AGENT_SYSTEM_PROMPT,
                target="user",
                context_policy=context_policy,
                request_extra=request_extra,
            )
            tools.append(
                task_tool(
                    [worker],
                    max_turns=DEFAULT_TASK_MAX_TURNS,
                    soft_turn_limit=DEFAULT_TASK_SOFT_TURN_LIMIT,
                )
            )
        elif capability == "skills":
            # Skills alter state initialization; they do not add a tool here.
            continue
    tools.extend(extra_tools)
    return tools


class AgentSession:
    """A context-managed agent runner that owns its toolsets' lifetimes.

    Enter the session to open every toolset (via one ``ExitStack``), gather
    their tools alongside the static tools, and build the ``Agent``. Call
    :meth:`run` while inside the ``with`` block; exit to close the toolsets.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        spec: AgentSpec | None = None,
        cwd: str | Path | None = None,
        target: str = "user",
        tools: Sequence[AgentTool] = (),
        toolsets: Sequence[Toolset] = (),
        skills: SkillConfig | None = None,
        mcp_servers: Sequence["MCPServerConfig"] = (),
        call_timeout: float = 60.0,
        connect: Callable[["MCPServerConfig"], "MCPConnection"] | None = None,
        context_policy: ContextPolicy | None = None,
        context_builder: ContextBuilderFn | None = None,
        hooks: HookMap | None = None,
        request_extra: Mapping[str, Any] | None = None,
        max_turns: int = 10,
        bash_exec_prefix: tuple[str, ...] = (),
        complete_fn: Callable[[LLMRequest], LLMResponse] | None = None,
    ) -> None:
        resolved_spec = spec or AgentSpec()
        self._provider = provider
        self._spec = resolved_spec
        self._name = resolved_spec.name
        self._role = resolved_spec.role
        self._system_prompt = (
            resolved_spec.system_prompt
            or compose_agent_system_prompt(
                bash="bash" in resolved_spec.capabilities,
                general_purpose="task" in resolved_spec.capabilities,
                skills="skills" in resolved_spec.capabilities,
                mcp=bool(mcp_servers),
            )
        )
        self._target = target
        self._static_tools = tuple(
            _capability_tools(
                provider,
                resolved_spec,
                cwd=cwd,
                extra_tools=tools,
                request_extra=request_extra,
                context_policy=context_policy,
                bash_exec_prefix=bash_exec_prefix,
            )
        )
        self._toolsets = (
            *tuple(toolsets),
            *tuple(
                MCPToolset(server, call_timeout=call_timeout, connect=connect)
                for server in mcp_servers
            ),
        )
        if "skills" in resolved_spec.capabilities:
            self._skills = skills or SkillConfig(cwd=str(cwd or "."))
        elif skills is not None:
            raise ValueError(
                "skills configuration requires 'skills' in AgentSpec.capabilities"
            )
        else:
            self._skills = None
        self._context_policy = context_policy
        self._context_builder = context_builder
        self._hooks = hooks or {}
        self._request_extra = request_extra
        self._max_turns = max_turns
        self._complete_fn = complete_fn

        self._stack: ExitStack | None = None
        self._agent: Agent | None = None

    @property
    def agent(self) -> Agent:
        """The built ``Agent`` — only available inside the ``with`` block."""

        if self._agent is None:
            raise RuntimeError(
                "AgentSession.agent is only available inside the context; "
                "enter the session first (use `with session as s: ...`)"
            )
        return self._agent

    def __enter__(self) -> "AgentSession":
        stack = ExitStack()
        tools: list[AgentTool] = list(self._static_tools)
        # Skills initialize state; they are not resources. Install them on the
        # built agent so `run` is a single `agent.run` for every capability mix.
        # MCP toolsets (opened below) are the only thing that needs the `with`
        # lifetime.
        init_state = (
            _skills_init_state(self._skills)
            if self._skills is not None and self._skills.enabled
            else None
        )
        try:
            for toolset in self._toolsets:
                opened = stack.enter_context(toolset)
                tools.extend(opened.tools())
            if self._spec.workflow == "single":
                agent_kwargs: dict[str, Any] = {}
                if self._complete_fn is not None:
                    agent_kwargs["complete_fn"] = self._complete_fn
                self._agent = make_llm_agent(
                    name=self._name,
                    provider=self._provider,
                    role=self._role,
                    tools=tools,
                    system_prompt=self._system_prompt,
                    target=self._target,
                    context_policy=self._context_policy,
                    context_builder=self._context_builder,
                    request_extra=self._request_extra,
                    init_state=init_state,
                    hooks=self._hooks,
                    **agent_kwargs,
                )
            else:
                if init_state is not None:
                    raise ValueError(
                        "the skills capability is currently supported only by workflow='single'"
                    )
                from horizon_harness.recipes.runtime import build_recipe_agent

                if self._context_builder is not None:
                    raise ValueError(
                        "custom context builders require workflow='single'"
                    )

                self._agent = build_recipe_agent(
                    spec=self._spec,
                    provider=self._provider,
                    tools=tools,
                    max_turns=self._max_turns,
                    context_policy=self._context_policy,
                    hooks=self._hooks,
                    request_extra=self._request_extra,
                )
        except BaseException:
            # A toolset that opened before the failure must still be closed.
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, *exc: object) -> None:
        self._agent = None
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.close()

    def run(
        self,
        task: ContentInput,
        *,
        max_turns: int | None = None,
        abort: AbortFlag = lambda: False,
    ) -> tuple[State, Iterator[Event]]:
        """Run the built agent on ``task``.

        A single ``agent.run``: any skills behavior comes from the initializer
        installed in :meth:`__enter__`, so this method no longer branches on
        skills. Returns ``(state, events)`` exactly like ``Agent.run`` —
        ``events`` is a lazy generator the caller iterates to advance the loop,
        so it must be consumed while the session (and any open toolset) is
        still entered.
        """

        agent = self.agent
        turns = self._max_turns if max_turns is None else max_turns
        return agent.run(task, max_turns=turns, abort=abort)


# --------------------------------------------------------------------------
# Prompt and identity defaults used by capability assembly.
# --------------------------------------------------------------------------

BASH_AGENT_SYSTEM_PROMPT = (
    "You are a tiny bash-use agent. Use the bash tool to satisfy the task "
    "— parallel tool calls are fine when the steps are independent, and "
    "set `attach` on the call when you want to surface an image file. "
    "After the tool_result, return a short final answer."
)
BASH_AGENT_DEFAULT_ROLE = (
    "Use bash for local commands, then summarize what you observed."
)
BASH_AGENT_DEFAULT_NAME = "bash_agent"

BASH_TASK_AGENT_DEFAULT_NAME = "bash_task_agent"
BASH_TASK_AGENT_DEFAULT_ROLE = (
    "Drive the task with bash for focused commands and delegate heavy "
    "exploration or multi-file reads to the bash worker via the task tool."
)
# Small addendum bolted onto the bash-agent system prompt so the parent
# inherits all of its phrasing and only learns the extra `task` affordance.
BASH_TASK_ADDENDUM = (
    "You also have a `task` tool that delegates a self-contained sub-task "
    "to a worker agent and returns its final summary. Use `task` aggressively "
    "as the default for any reasonably self-contained sub-task; delegate "
    "multiple independent sub-tasks separately instead of handling them "
    "yourself with `bash`. Use direct `bash` only when delegation would "
    "clearly be unnecessary or when you must run the final local command "
    "yourself. When uncertain, delegate."
)
BASH_TASK_AGENT_SYSTEM_PROMPT = BASH_AGENT_SYSTEM_PROMPT + "\n\n" + BASH_TASK_ADDENDUM

GENERAL_PURPOSE_AGENT_DEFAULT_NAME = "general-purpose"
GENERAL_PURPOSE_AGENT_DEFAULT_ROLE = (
    "Handle a delegated sub-task end to end with bash — investigate, make "
    "the needed edits, run focused checks — then return a short summary."
)
GENERAL_PURPOSE_AGENT_SYSTEM_PROMPT = (
    "You are a general-purpose bash-using sub-agent. Satisfy the delegated "
    "task by running bash commands (parallel reads when the steps are "
    "independent): investigate the workspace, make any edits the task calls "
    "for, and run the focused checks needed to verify them. Prefer narrow "
    "tools (`grep -n`, `head`/`tail`, `sed -n 'A,Bp'`, `find`) over dumping "
    "whole files. When done, return a short final summary the parent can act "
    "on. Stop as soon as the task is satisfied."
)
DEFAULT_TASK_MAX_TURNS = 70
DEFAULT_TASK_SOFT_TURN_LIMIT = 60

DEFAULT_AGENT_NAME = "agent"

SKILLS_ADDENDUM = (
    "You also have access to a library of skills. When a skill fits the task, "
    "read its SKILL.md first, then run its scripts via bash. Prefer a matching "
    "skill over improvising, and work from the evidence it produces."
)

MCP_ADDENDUM = (
    "You also have tools provided by one or more connected MCP servers (their "
    "names are prefixed with the server name). Use them when they fit the task, "
    "then return a short final answer."
)


def compose_agent_system_prompt(
    *, bash: bool, general_purpose: bool, skills: bool, mcp: bool
) -> str:
    """Build a system prompt by appending capability fragments to the base.

    The base is always the bash-agent prompt; ``general_purpose``/``skills``/
    ``mcp`` each append their fragment in that fixed order. Callers that pass an
    explicit ``system_prompt`` bypass this entirely. When ``bash`` is false the
    base still reads as the bash prompt, so a bash-less agent should supply its
    own ``system_prompt`` (see the design spec's edge-case note).
    """

    del bash  # base is unconditional today; kept for signature symmetry
    parts = [BASH_AGENT_SYSTEM_PROMPT]
    if general_purpose:
        parts.append(BASH_TASK_ADDENDUM)
    if skills:
        parts.append(SKILLS_ADDENDUM)
    if mcp:
        parts.append(MCP_ADDENDUM)
    return "\n\n".join(parts)
