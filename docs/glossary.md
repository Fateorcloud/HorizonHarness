# Glossary

General vocabulary used across the repo. The message-protocol terms —
`Message`, `LLMMessage`, `RuntimeMessage`, content blocks, provider adapters,
and message sidecars — are defined once in [`CONTEXT.md`](../CONTEXT.md); this
file does not repeat them.

## Agent

A system that uses a model to decide what to do next, often across multiple steps.

## Agent Loop

The repeated control flow where the system observes state, asks a model for the next action, executes that action, and records the result.

## Event

A typed record in the run trace. In the current core, message events wrap
transcript messages, while lifecycle, model request/response, and tool
execution events record the rest of the run.

## Tool

A callable capability exposed to the agent, such as reading a file, searching
data, calling an API, or running a calculation. In this repo, shared tool
values live in `horizon_harness.tools`; each runtime owns its own dispatch
semantics.

## State

The data the system carries between steps in an agent loop.

## Continue Session

`Agent.continue_session(state, followup)` appends a follow-up to a caller-held
in-memory `State` and continues the same message and event history. It does not
persist a checkpoint and is not process crash recovery.

## Chain Context Continuation

An Eval Chain payload that carries active messages, task, and data between
ordered benchmark instances. It deliberately excludes the complete event
stream, pending tool state, and execution cursor.

## Batch Reattachment

A fresh control process reconnecting to external container jobs to poll or
collect their results. `submit_dataset` / `reconcile_dataset` use this meaning;
they do not resume an interrupted Agent Loop.

## Resume Run

The reserved name for a future process-level crash recovery operation backed
by persistent checkpoints, explicit commit boundaries, and side-effect
handling. `resume_run` is not implemented in the current release.

## Memory

Information preserved beyond a single immediate step or turn.

## Compressed Context and Recall

Compression changes the active model view, not the append-only transcript.
Rule-based tool previews cite original result-message indices so an explicitly
registered `recall` tool can retrieve details absent from the preview. A preview
is not evidence that omitted facts never existed. Registration remains the
caller's responsibility; a default AgentSession does not enable recall.
Retrieval is bounded and may truncate long messages. This mechanism does not
guarantee that a model will retrieve or correctly use the information.

## Evaluation

A repeatable check that helps compare agent behavior, correctness, reliability, or usability.

## Reference Architecture

An external or internal architecture studied before deciding what
HorizonHarness should borrow or avoid.
