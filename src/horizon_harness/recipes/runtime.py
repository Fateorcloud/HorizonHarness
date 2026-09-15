"""Default Recipe assembly used by :class:`AgentSession`.

Recipes orchestrate ordinary ``Agent`` objects. They do not introduce a second
model/tool loop: every participant is still driven through ``core.run`` by the
recipe primitives in this package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from horizon_harness.context_view import ContextPolicy
from horizon_harness.core import Agent
from horizon_harness.hooks import HookContext, HookMap, HookPoint
from horizon_harness.llm import Provider
from horizon_harness.llm_agent import make_llm_agent
from horizon_harness.messages import Message, assistant_message, text_of
from horizon_harness.spec import AgentSpec
from horizon_harness.tools import AgentTool

from .base import WorkflowResult
from .goal_checks import model_declared_check, update_goal_tool
from .goal_loop import GoalBudgets, run_goal_loop
from .parallel import make_aggregator_agent, run_parallel
from .pdr import make_distiller_agent, run_pdr
from .planner_executor import (
    make_executor_agent,
    make_planner_agent,
    run_planner_executor,
)
from .reflection import make_critic_agent, make_generator_agent, run_reflection
from .routing import Route, make_router_agent, run_routing
from .sequential import run_chain
from .trace import workflow_steps_breakdown


def _participant(
    provider: Provider,
    *,
    name: str,
    role: str,
    system_prompt: str,
    tools: Sequence[AgentTool],
    context_policy: ContextPolicy | None,
    request_extra: Mapping[str, Any] | None,
) -> Agent:
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        tools=tools,
        system_prompt=system_prompt,
        target="user",
        context_policy=context_policy,
        request_extra=request_extra,
    )


def _task_text(visible: Sequence[Message]) -> str:
    for message in reversed(visible):
        if message.kind == "task":
            return text_of(message.content)
    return text_of(visible[-1].content) if visible else ""


def run_recipe(
    spec: AgentSpec,
    provider: Provider,
    tools: Sequence[AgentTool],
    task: str,
    *,
    max_turns: int,
    context_policy: ContextPolicy | None = None,
    request_extra: Mapping[str, Any] | None = None,
) -> WorkflowResult:
    """Run the default implementation for one non-``single`` workflow."""

    def worker(name: str, role: str, prompt: str) -> Agent:
        return _participant(
            provider,
            name=name,
            role=role,
            system_prompt=prompt,
            tools=tools,
            context_policy=context_policy,
            request_extra=request_extra,
        )

    workflow = spec.workflow
    if workflow == "chain":
        stages = [
            worker(
                "analyst",
                "Extract the constraints and a concrete approach.",
                "Analyze the task and return a concise, actionable solution outline.",
            ),
            worker(
                "solver",
                spec.role or "Execute the task using the prior analysis.",
                spec.system_prompt
                or "Complete the task using the previous stage's analysis and verify the result.",
            ),
        ]
        return run_chain(stages, task, max_turns=max_turns)

    if workflow == "planner_executor":
        return run_planner_executor(
            make_planner_agent(provider, request_extra=request_extra),
            make_executor_agent(provider, tools=tools, request_extra=request_extra),
            task,
            planner_max_turns=max_turns,
            executor_max_turns=max_turns,
        )

    if workflow == "reflection":
        return run_reflection(
            make_generator_agent(provider, tools=tools, request_extra=request_extra),
            make_critic_agent(provider, request_extra=request_extra),
            task,
            max_rounds=1,
            generator_max_turns=max_turns,
            critic_max_turns=max_turns,
        )

    if workflow == "routing":
        specialist = worker(
            "general",
            spec.role or "Handle the selected task end to end.",
            spec.system_prompt or "Complete the routed task and return the result.",
        )
        routes = [Route("general", specialist, specialist.role)]
        router = make_router_agent(provider, routes, request_extra=request_extra)
        return run_routing(
            router,
            routes,
            task,
            router_max_turns=max_turns,
            worker_max_turns=max_turns,
            default="general",
        )

    if workflow == "parallel":
        workers = [
            worker(
                f"worker_{index}",
                "Produce an independent solution.",
                "Solve the task independently and return a complete answer.",
            )
            for index in range(2)
        ]
        return run_parallel(
            workers,
            task,
            aggregator=make_aggregator_agent(provider, request_extra=request_extra),
            worker_max_turns=max_turns,
            aggregator_max_turns=max_turns,
        )

    if workflow == "goal_loop":
        solver = _participant(
            provider,
            name=spec.name,
            role=spec.role,
            system_prompt=spec.system_prompt
            or "Work until the task is complete, then declare status with update_goal.",
            tools=[*tools, update_goal_tool()],
            context_policy=context_policy,
            request_extra=request_extra,
        )
        result = run_goal_loop(
            solver,
            task,
            check=model_declared_check,
            budgets=GoalBudgets(max_turns=2),
            inner_max_turns=max_turns,
        )
        return WorkflowResult(output=result.output, steps=result.steps)

    if workflow == "pdr":
        workers = [
            worker(
                f"attempt_{index}",
                "Produce an independent attempt.",
                "Attempt the task independently and report concrete progress.",
            )
            for index in range(2)
        ]
        return run_pdr(
            workers,
            make_distiller_agent(provider, request_extra=request_extra),
            task,
            rounds=1,
            finalizer=worker(
                "finalizer",
                spec.role or "Produce the final result.",
                spec.system_prompt or "Complete the task using the distilled findings.",
            ),
            worker_max_turns=max_turns,
            distiller_max_turns=max_turns,
            finalizer_max_turns=max_turns,
        )

    raise ValueError(f"workflow {workflow!r} is not a Recipe")


def build_recipe_agent(
    *,
    spec: AgentSpec,
    provider: Provider,
    tools: Sequence[AgentTool],
    max_turns: int,
    context_policy: ContextPolicy | None = None,
    hooks: HookMap | None = None,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Wrap a Recipe behind the ordinary ``Agent.run`` interface."""

    last_result: WorkflowResult | None = None

    def generate(visible: list[Message]) -> Message:
        nonlocal last_result
        last_result = run_recipe(
            spec,
            provider,
            tools,
            _task_text(visible),
            max_turns=max_turns,
            context_policy=context_policy,
            request_extra=request_extra,
        )
        return assistant_message(
            last_result.output,
            sender=spec.name,
            target="user",
            kind="final",
        )

    def record_recipe(ctx: HookContext) -> None:
        if last_result is not None:
            ctx.state.data["recipe"] = workflow_steps_breakdown(
                last_result, spec.workflow
            )

    merged_hooks: HookMap = {
        point: list(callbacks) for point, callbacks in (hooks or {}).items()
    }
    merged_hooks.setdefault(HookPoint.SESSION_END, []).append(record_recipe)
    return Agent(
        name=spec.name,
        generate=generate,
        role=spec.role,
        system_prompt=spec.system_prompt,
        hooks=merged_hooks,
    )


__all__ = ["build_recipe_agent", "run_recipe"]
