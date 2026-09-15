# Agent Assembly

`AgentSession` is the high-level assembly entry point for HorizonHarness. It
combines one `AgentSpec`, the selected capabilities, and an optional workflow
recipe around the same core Agent Loop.

## Standard path

The default benchmark and the shortest learning path use one ordinary agent
with `bash`, `read`, and `task` capabilities:

```python
from horizon_harness import AgentSession, AgentSpec, Provider

provider = Provider(id="fake", api="fake", model="fake-model")
spec = AgentSpec(name="solver", capabilities=("bash", "read", "task"), workflow="single")

with AgentSession(provider, spec=spec, cwd=".") as session:
    state, events = session.run("Inspect the project and summarize it.")
    for _ in events:
        pass
```

The event iterator is lazy and must be consumed while the session is entered.
`State` keeps the append-only messages and `Event` history for the run.

## Two independent axes

- `capabilities` selects tools: `bash`, `read`, `edit`, `task`, and `skills`.
- `workflow` selects orchestration: `single`, `chain`, `planner_executor`,
  `reflection`, `routing`, `parallel`, `goal_loop`, or `pdr`.

These axes are deliberately independent. A workflow recipe composes ordinary
agents through the same loop; it is not a second family of Agent variants.
Recipes live under `horizon_harness.recipes` and are explicit imports.

## Resource-owning capabilities

MCP connections are opened and closed by `AgentSession`:

```python
from horizon_harness import AgentSession, AgentSpec
from horizon_harness.mcp import MCPServerConfig

fs = MCPServerConfig.stdio("fs", "npx", "-y", "@modelcontextprotocol/server-filesystem", ".")
spec = AgentSpec(capabilities=("bash", "read"), workflow="single")

with AgentSession(provider, spec=spec, mcp_servers=[fs]) as session:
    state, events = session.run("Inspect the workspace.")
    for _ in events:
        pass
```

Skills use the same state-initializer path. Enable the `skills` capability and
pass a `SkillConfig` when discovery or preload behavior needs customization.

## Continuation boundary

`Agent.continue_session(state, followup)` appends a follow-up to the caller-held
in-memory `State`. It preserves the existing message and event history; it is
not process-level checkpoint recovery. The reserved future name
`resume_run(run_id)` is not implemented.

## Package layout

- `starter.py`: `AgentSession`, capability assembly, and prompt defaults.
- `toolsets.py`: the `Toolset` protocol and MCP adapter.
- `horizon_harness.recipes`: workflow implementations over the common loop.

Advanced modules are intentionally imported from their owning subpackages.
The root package exports only the stable runtime concepts documented in the
top-level README.
