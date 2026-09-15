# Memory

Implementation guide for `src/horizon_harness/memory/`. Code and
`tests/unit/test_memory.py` and `tests/unit/test_indexed_memory.py` remain the source of truth for behavior; this file
holds the decisions that are not visible in either.

## The interface

```text
Memory
  initial(ctx)  -> tuple[Message, ...]    at SESSION_START
  tools(ctx)    -> tuple[AgentTool, ...]  at assembly
  finish(ctx)   -> None                   at SESSION_END
```

That is exactly what the core hook points can drive today. Query-shaped recall
is supplied as ordinary `memory_search` / `memory_read` tools by
`IndexedMemory`; the model decides when to call them. Per-turn observation
would still need a corresponding hook point rather than a no-op seam.

Memory stays outside the core: it injects ordinary `Message` values, exposes
ordinary `AgentTool`s, and binds through the normal `HookMap`. Nothing in
`Message`, the context view, or the provider adapters knows memory exists.

Bind before building the agent, then pass `binding.tools` and `binding.hooks`
into `make_llm_agent` — the same path MCP toolsets use.

Injected memory is a `RuntimeMessage` with `sender="memory"`,
`kind="context"` (`memory_context_message(...)`). Prompt-only injection is a
special case and must still be visible in the trace — hidden prompt mutation
makes it impossible to explain why the model knew something.

## Quality, not recall

The durable lesson is not to maximize recall:

- Do not save secrets, large raw outputs, generic advice, current-task state, or
  anything cheaper to read from code, docs, or git.
- User messages, corrections, accepted workflows, and tool evidence outrank
  assistant-authored summaries.
- Memory naming current files, flags, or repo state is a dated observation;
  verify it against the checkout when that is cheap. Workspace facts always
  outrank stale memory.
- "No memory update" is a successful outcome.
- Do not assume vector or hybrid ranking is better. `FilesystemMemory` remains
  the inspectable default. The optional indexed backend exists for a frozen
  keyword-versus-hybrid comparison and must report a negative result when it
  does not improve task outcomes.

## IndexedMemory: sourced entries and retrieval

`IndexedMemory` is a separate SQLite implementation. It stores immutable
`fact`, `decision`, and `experience` entries; an experience must name its
condition, method, and observed outcome. Every candidate must reference a
source registered for the same task/run. Repeating `finish_task` for the same
task revision is idempotent. A new same-scope entry may explicitly supersede an
active old entry; cross-scope replacement is rejected, and conflict groups stay
visible instead of selecting truth by recency.

`finish()` intentionally learns nothing at generic `SESSION_END`, because that
event also covers pauses and budget stops. `TaskSession` can attach a
`ModelMemoryExtractor`; at explicit `completed`/`failed` finalization it passes
bounded stable-reference messages, Workspace state, artifact metadata and
verifier records to one extraction call, registers those sources, validates at
most ten returned candidates, then writes them transactionally. Extraction
failure is returned separately and does not turn a valid code task into a
failure. `finish_task(...)` remains the lower-level fixture/import seam for
already prepared candidates; it is not evidence that extraction ran.

Keyword search uses SQLite FTS5 after deterministic camelCase, snake_case and
Chinese-bigram preprocessing. Hybrid mode filters by the same scope, retrieves
up to twenty keyword and semantic candidates, and joins them with RRF (`k=60`,
stable id tie-break). It never silently degrades: constructing hybrid mode
without an embedder fails, and model revision or vector dimension mismatches
require an explicit rebuild. Default results exclude superseded entries and
are capped by count and an estimated 2,000-token payload.

The real embedder is lazy, CPU-only
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Install the
optional dependency and pass an exact model revision:

```bash
uv sync --extra memory
uv run --extra memory python -m runs.memory_validation --db evals/out/memory.sqlite3 \
  run --mode hybrid --query "test command" --scope /absolute/repository/path \
  --revision <hugging-face-commit> --output evals/out/memory-hybrid.json
```

The output records actual revision, dimension, dependency version and elapsed
time. `memory_validation prepare` is explicitly fixture-only; real task
extraction uses `runs.task_workspace --extract-memory-on-complete`. A
fake-vector unit test covers ranking offline; it is not a substitute for the
embedding smoke or task-level memory benefit.

## FilesystemMemory layout and guarantees

Memory root defaults to `~/.simple/memory`; pass `root=...` to place it under an
eval output dir, a mounted container dir, or a workshop dir. Each namespace is
`{root}/{memory_name}` and holds:

| File | Owner | Role |
| --- | --- | --- |
| `MEMORY.md` | the distiller model | the single durable handbook, rewritten whole |
| `memory_summary.md` | rebuilt from `INDEX.md` unless the model supplies one | cold-start navigation |
| `INDEX.md` | `finish(...)` | routes to per-run evidence |
| `runs/{run_id}/` | host-written | raw evidence + `artifacts/` |

Decisions worth knowing:

- **One writer per root.** `finish(...)` serializes the whole
  read-distill-commit operation with an inter-process lock on the root — the
  model call included, because `MEMORY.md` is a full rewrite and the distiller
  may pick the namespace only after reading prior memory. Atomic temp-file
  replacement gives per-file atomicity; it is not a substitute for this
  logical single-writer boundary. Child-only namespace mounts share the root
  `.memory-lock/`.
- **The model owns merging.** The distiller sees the current `MEMORY.md` and
  returns the complete updated file, doing its own dedup, rewrite, and pruning.
  `finish(...)` keeps no section or promotion logic. Determinism lives only in
  the guardrails: an empty rewrite keeps the prior handbook, and one that is
  oversized, structurally empty, or would erase every lesson is rejected, with
  a `memory_error.md` marker recording the skip.
- **Failure is never fatal.** Bounded run evidence and a minimal `summary.md`
  are still written when distillation fails, so `INDEX.md` never points at a
  missing summary. A memory failure must not fail an otherwise valid run.
- **Limits are small on purpose.** 64 runs and 128 MiB per namespace, 128
  namespaces per root, oldest evidence pruned after a write. All in
  `FilesystemMemoryLimits`.
- **Run ids must be distinct.** `runs/{run_id}` falls back to `session_id` then
  a timestamp; a complete existing run with the same id makes `finish(...)` a
  no-op.

## Recall is model-driven

`initial(...)` injects a short policy block naming the memory path (or, when no
namespace is known, the root plus at most 64 namespace names). The model then
reads memory through the ordinary `read`/`bash` tools. The runtime does **not**
prefetch or inject matching lines before a model request. Keep it that way for
filesystem memory; a runtime-retrieval shape (vector snippets) would be a
different `Memory` implementation, not a change to this one.

## Keeping it domain-neutral

The mechanism came from SWE-bench-shaped work. Keep the generic shape:

- Domain outputs go under `artifacts/` — not a hard-coded `patch.diff`. SWE
  stores `submission.txt`; another domain stores a JSON or a report.
- Index fields stay generic: `Summary`, `Scope`, `Signals`, `Keywords`,
  `Artifacts`.
- Benchmark labels and scoring outcomes must never leak into future runs.

## Evals integration

Container runs opt in through `HORIZON_MEMORY_HOME` (plus `HORIZON_MEMORY_NAME` /
`HORIZON_MEMORY_RUN_ID`); `LocalDockerBackend(memory_home=...)` bind-mounts the host
directory and sets them. Memory itself never imports Docker or evals — assembly
code passes it a local path. A suite's optional `memory_artifacts(...)` hook
supplies that run's durable products, captured inside `memory.finish` at
`SESSION_END` while the workspace is still intact.
