# HorizonHarness

Long-horizon agent runtime and evaluation harness.

HorizonHarness turns multi-turn model calls and tool use into an explicit,
inspectable execution process. The project is intentionally small enough to
teach: one Agent Loop, typed State/Message/Event values, an explicit Tool
Runtime, context projection and compression, and trace/evaluation tooling.

## Runtime shape

```text
AgentSession(AgentSpec)
        |
        +-- Agent Loop: model -> tool -> observation -> next turn
        +-- State / Message / Event
        +-- Context View and compression
        +-- Trace and cost accounting
        +-- Eval runners and recipes
        |
        +-- Provider adapters and external tools
```

The public root package deliberately exports only the stable assembly concepts:
`Agent`, `State`, `Message`, `Event`, `AgentTool`, `ToolResult`, `AgentSpec`,
`AgentSession`, `Provider`, and `run`. Message constructors, tools, context
policies, recipes, tracing, memory, MCP, and benchmark helpers are imported from
their owning subpackages.

## Standard path

The documented path is one ordinary loop with `bash`, `read`, and `task`
capabilities. Capabilities and workflow are independent dimensions; there is no
combined selector string.

```python
from horizon_harness import AgentSession, AgentSpec, Provider

provider = Provider(id="fake", api="fake", model="fake-model")
spec = AgentSpec(
    name="solver",
    capabilities=("bash", "read", "task"),
    workflow="single",
)

with AgentSession(provider, spec=spec, cwd=".") as session:
    state, events = session.run("Inspect the repository and report the result.")
    tuple(events)
```

Run the deterministic demo without credentials:

```bash
uv sync
bash runs/demos/run_bash_agent_demo.sh
```

Real OpenAI-compatible providers use the root `.env` file. For the history
reuse experiments, use `.env.history-reuse.glm` first and only fall back to
`.env.history-reuse.deepseek` when GLM is unavailable. Keep credentials out of
the repository and record the selected endpoint in the run manifest.

## What the harness makes explicit

- `State` is append-only runtime state; `Event` is the typed execution trace.
- `continue_session(state, followup)` appends a follow-up to caller-held,
  in-memory state. It is session continuation, not crash recovery.
- `chain context continuation` passes active messages, task, and data between
  ordered evaluation instances. It does not restore an event stream, pending
  tool call, or execution cursor.
- `batch reattachment` means a fresh process polls and collects external
  container jobs. It does not resume an Agent Loop.
- `resume_run` is reserved for a future checkpoint-based crash recovery API;
  it is intentionally not implemented in this phase.
- `horizon_harness.workspace.TaskSession` is the narrower durable continuation
  API. It persists canonical model decisions and can recover an unfinished
  batch of controlled `edit` calls in a new process before the next model call.
  Other side-effecting tools have no general exactly-once guarantee and must
  provide their own reconciliation contract before durable recovery is safe.

The Observatory trace viewer is the only GUI in this phase. No new TUI or chat
application is part of the runtime refactor.

Run a key-free checkpoint smoke through the same CLI used for live tasks:

```bash
uv run python -m runs.task_workspace --fake --store-dir evals/out/workspace-smoke \
  start "Return a short answer." --task-id smoke --pause-after-turns 1
uv run python -m runs.task_workspace --store-dir evals/out/workspace-smoke inspect smoke
uv run python -m runs.task_workspace --fake --store-dir evals/out/workspace-smoke \
  resume smoke --pause-after-turns 1
```

`runs.workspace_memory_eval plan` freezes the dedicated four-task S/W/K/Y
acceptance matrix, implementation and oracle hashes, and token budgets without
calling a model. Its separate `run` subcommand is the paid path; it uses safe
checkpoint continuations and the official SWE-bench verifier while keeping
verifier rows outside Agent-visible workspaces.

## Evaluation and benchmark plan

Evaluation code is grouped by purpose:

- `src/horizon_harness/recipes/`: workflow recipes that reuse the same loop.
- `src/horizon_harness/evals/`: container, chain, memory, and benchmark seams.
- `studio/trace-viewer/`: HorizonHarness Observatory.
- `src/horizon_harness/evals/benchmark.py`: offline model matrix and run manifest.

The frozen comparison matrix is `gpt-5.6-sol` via `openai-responses` with
`reasoning=high`, GLM-5.2 free route (`z-ai/glm-5.2:free`) via `openai-chat`, and
`deepseek-v4-flash` via `openai-chat`. Formal Terminal-Bench 2.1 and the fixed
SWE-bench Pro auxiliary set are not run by installation or CI. The unified
`runs.run_bench` entry point writes a Run Manifest before it invokes an existing
benchmark adapter, then copies that frozen launch record into the adapter's run
directory. API preflight plus the existing CI gate runtime reliability; the
18-case list is coverage guidance, not a second benchmark runner or a paid-run
gate.

No HorizonHarness score is published until the refactored runtime has been
re-run from a frozen commit. Historical scores from the previous project name
are not inherited.

## Development

Commands use the pinned `uv` environment:

```bash
uv run python -m unittest discover -s tests/unit
uv run python -m scripts.lint_docs
bash runs/dev/run_ci.sh
```

The project is licensed under Apache-2.0. Existing attribution and Git history
are retained.
