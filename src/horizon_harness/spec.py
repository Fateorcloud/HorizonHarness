"""Stable data contract for assembling HorizonHarness agents."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

Capability = Literal["bash", "read", "edit", "task", "skills"]
Workflow = Literal[
    "single",
    "chain",
    "planner_executor",
    "reflection",
    "routing",
    "parallel",
    "goal_loop",
    "pdr",
]

CAPABILITIES: tuple[Capability, ...] = (
    "bash",
    "read",
    "edit",
    "task",
    "skills",
)
WORKFLOWS: tuple[Workflow, ...] = (
    "single",
    "chain",
    "planner_executor",
    "reflection",
    "routing",
    "parallel",
    "goal_loop",
    "pdr",
)
DEFAULT_CAPABILITIES: tuple[Capability, ...] = ("bash", "read", "task")
CAPABILITIES_ENV = "HORIZON_CAPABILITIES"
WORKFLOW_ENV = "HORIZON_WORKFLOW"


@dataclass(frozen=True)
class AgentSpec:
    """Declarative agent identity plus two independent assembly axes.

    ``capabilities`` selects what an agent can do. ``workflow`` selects how
    one or more ordinary agents are orchestrated. A workflow never changes the
    core agent loop used by its participants.
    """

    name: str = "agent"
    role: str = ""
    system_prompt: str = ""
    capabilities: tuple[Capability, ...] = DEFAULT_CAPABILITIES
    workflow: Workflow = "single"

    def __post_init__(self) -> None:
        requested = tuple(self.capabilities)
        unknown = tuple(value for value in requested if value not in CAPABILITIES)
        if unknown:
            raise ValueError(
                f"unsupported capability {unknown[0]!r}; expected one of {CAPABILITIES}"
            )
        if len(set(requested)) != len(requested):
            raise ValueError("capability names must not be repeated")
        if self.workflow not in WORKFLOWS:
            raise ValueError(
                f"unsupported workflow {self.workflow!r}; expected one of {WORKFLOWS}"
            )
        if not self.name.strip():
            raise ValueError("agent name must not be empty")
        object.__setattr__(self, "capabilities", requested)


def agent_spec_from_env(
    base: AgentSpec | None = None,
    env: Mapping[str, str] | None = None,
) -> AgentSpec:
    """Apply the two process-level assembly selectors to an ``AgentSpec``."""

    source = env if env is not None else os.environ  # env-ok: assembly selector owner
    if "AGENT_FLAVOR" in source:
        raise ValueError(
            "AGENT_FLAVOR was removed; set HORIZON_CAPABILITIES and "
            "HORIZON_WORKFLOW independently"
        )
    baseline = base or AgentSpec()
    capabilities = baseline.capabilities
    if CAPABILITIES_ENV in source:
        capabilities = cast(
            tuple[Capability, ...],
            tuple(
                part.strip().lower()
                for part in source[CAPABILITIES_ENV].split(",")
                if part.strip()
            ),
        )
    workflow = cast(
        Workflow,
        source.get(WORKFLOW_ENV, baseline.workflow).strip().lower()
        or baseline.workflow,
    )
    return AgentSpec(
        name=baseline.name,
        role=baseline.role,
        system_prompt=baseline.system_prompt,
        capabilities=capabilities,
        workflow=workflow,
    )


__all__ = [
    "AgentSpec",
    "Capability",
    "Workflow",
    "CAPABILITIES",
    "WORKFLOWS",
    "DEFAULT_CAPABILITIES",
    "CAPABILITIES_ENV",
    "WORKFLOW_ENV",
    "agent_spec_from_env",
]
