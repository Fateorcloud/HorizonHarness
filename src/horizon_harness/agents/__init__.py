"""Agent assembly and resource-bearing toolsets.

``AgentSession`` is the only high-level assembly entry point.  Configure its
capabilities and recipe with :class:`horizon_harness.AgentSpec`.
"""

from .starter import (
    BASH_AGENT_DEFAULT_NAME,
    BASH_AGENT_DEFAULT_ROLE,
    BASH_AGENT_SYSTEM_PROMPT,
    BASH_TASK_AGENT_DEFAULT_NAME,
    BASH_TASK_AGENT_DEFAULT_ROLE,
    BASH_TASK_AGENT_SYSTEM_PROMPT,
    BASH_TASK_ADDENDUM,
    DEFAULT_AGENT_NAME,
    DEFAULT_TASK_MAX_TURNS,
    DEFAULT_TASK_SOFT_TURN_LIMIT,
    GENERAL_PURPOSE_AGENT_DEFAULT_NAME,
    GENERAL_PURPOSE_AGENT_DEFAULT_ROLE,
    GENERAL_PURPOSE_AGENT_SYSTEM_PROMPT,
    MCP_ADDENDUM,
    SKILLS_ADDENDUM,
    AgentSession,
    SkillConfig,
    compose_agent_system_prompt,
)
from .toolsets import MCPToolset, Toolset

__all__ = [
    "AgentSession",
    "SkillConfig",
    "Toolset",
    "MCPToolset",
    "compose_agent_system_prompt",
    "DEFAULT_AGENT_NAME",
    "SKILLS_ADDENDUM",
    "MCP_ADDENDUM",
    "BASH_AGENT_SYSTEM_PROMPT",
    "BASH_AGENT_DEFAULT_ROLE",
    "BASH_AGENT_DEFAULT_NAME",
    "BASH_TASK_AGENT_SYSTEM_PROMPT",
    "BASH_TASK_AGENT_DEFAULT_ROLE",
    "BASH_TASK_AGENT_DEFAULT_NAME",
    "BASH_TASK_ADDENDUM",
    "GENERAL_PURPOSE_AGENT_DEFAULT_NAME",
    "GENERAL_PURPOSE_AGENT_DEFAULT_ROLE",
    "GENERAL_PURPOSE_AGENT_SYSTEM_PROMPT",
    "DEFAULT_TASK_MAX_TURNS",
    "DEFAULT_TASK_SOFT_TURN_LIMIT",
]
