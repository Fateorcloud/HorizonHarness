# Task Workspace

`horizon_harness.workspace.TaskSession` adds durable continuation for one
`workflow="single"` text coding task while reusing `AgentSession` and the
ordinary Agent Loop. The source of truth is the versioned task record under the
configured store directory; `State.data` is never serialized.

## Safety contract

A checkpoint is resumable only after a complete `TurnEndEvent`, when every tool
call in that turn has a recorded result. Before a turn starts, the store writes
`uncommitted.json`; a clean checkpoint commit atomically replaces its JSON and
then clears that marker. If a process exits during the turn or persistence
fails, `resume()` refuses automatic continuation because a tool may have made
an unconfirmed side effect.

This boundary deliberately does not restore a running command, provider-private
state, hidden reasoning, or an execution cursor. It provides safe checkpoint
continuation, not `resume_run`, arbitrary crash recovery, exactly-once external
effects, or rollback.

One file lock covers an entire start/resume run. A checkpoint records the code
directory plus Git HEAD identity and a credential-free provider/configuration
fingerprint. Resume rejects mismatches. Registered artifacts are relative to
the code directory and carry SHA-256 values. If an artifact changed externally,
the file is preserved, its verification references are cleared, and the task
is returned for verification.

## Task truth and model context

`WorkspaceRecord` keeps the original goal, immutable caller constraints,
acceptance text, stable steps, decisions, artifacts, blockers, next action and
task status. Progress is derived from step status. A model may update those
fields through sequential `workspace_update`, using an expected revision, but
only the host can append independent verifier results. `completed` is therefore
a workflow status, not proof that tests passed.

The optional core context-builder seam gives Workspace ownership of the whole
projection. Its view pins the original task and Workspace state, retains the
latest memory policy when enabled, and keeps the two most recent complete
assistant/tool-result turns. Older message references remain readable through
`history_read`. Budget trimming removes a whole assistant/tool-result turn and
updates included/archive references together; it never removes one side of a
tool exchange. The estimate includes the actual bound `bash`, `read`, `edit`,
Workspace, history and memory schemas plus the system prompt. The effective
input cap is the smaller of the configured 12,000-token limit and the provider
context window after the output reserve and safety buffer. Context is recomputed
from selected messages, not a historical full-request usage baseline that includes
archived messages. If the required
state plus latest complete turn cannot fit, the builder fails instead of
truncating the contract.

The formal S baseline can resume the same checkpoint with a one-shot bounded
summary representation. That path keeps `history_read` but does not expose
`workspace_update`; its trace identifies the builder as `summary`, so it cannot
be confused with a Workspace continuation.

An opt-in `combined` representation composes the two roles: Workspace remains
the persistent authority for current constraints, status and next action; a
bounded handoff summary supplies narrative context with an explicit source
revision; `workspace_read` exposes older steps, decisions, artifacts and
blockers on demand. A stale summary is labelled and cannot override current
state. The combined H3 recovery case passed its official verifier, which proves
one real continuation is feasible, not a general efficiency improvement.

## CLI

For combined handoff, use `TaskSession(context_representation="combined",
continuation_summary=..., summary_revision=...)`. Persistent Workspace remains
the task-state owner; only current constraints, acceptance, status and next action
are injected alongside the original goal, historical summary and recent turns.
Full steps, decisions, artifacts and blockers are read through `workspace_read`
(three entries per page). `workspace_update` still persists changes.

The summary's source revision is explicit. A future revision is rejected; an older
summary is labelled stale and cannot override current state. On conflict the agent
is instructed to read current details and source evidence. This is an instruction
and traceable contract, not a guarantee of model compliance. Summaries are not
automatically regenerated each turn. Existing S/W modes remain available.

The CLI accepts `--handoff` before the subcommand, pointing to a JSON file with
`text` and `source_revision`. Pass the same file to a new process's `resume` call;
keep it with task artifacts. This opt-in does not change the default CLI mode.

The fake path needs no credentials and proves disk continuation across process
invocations:

```bash
uv run python -m runs.task_workspace --fake --store-dir evals/out/workspace-smoke \
  start "Return a short answer." --task-id smoke --pause-after-turns 1
uv run python -m runs.task_workspace --store-dir evals/out/workspace-smoke inspect smoke
uv run python -m runs.task_workspace --fake --store-dir evals/out/workspace-smoke \
  resume smoke --pause-after-turns 1
```

Omit `--fake` for the root `.env` provider. `start` and `resume` write segmented
Trace JSONL and checkpoint messages below `<store-dir>/<task-id>/`; `inspect` is
read-only.

For a real terminal task, indexed memory can be attached and one extraction
call can be requested explicitly. Extraction runs only after `completed` or
`failed`, receives bounded task messages/Workspace/artifact/verifier evidence,
and can only write candidates whose stable source reference belongs to that
task. Pauses and budget stops do not extract:

```bash
uv run python -m runs.task_workspace \
  --dotenv .env.history-reuse.glm \
  --memory-db evals/out/task-memory.sqlite3 \
  --memory-scope /absolute/repository/path \
  --extract-memory-on-complete \
  start "Complete the repository task."
```
