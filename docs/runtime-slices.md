# Durable Runtime Status

This document separates code presence, passing tests, and demonstrated runtime
contracts. Code and tests remain the source of truth.

## Current architecture

```text
TaskSession
  validates durable history, binds provider/tools/reconciliation, rebuilds State
       |
       v
core.run()
  owns pending-batch, tool dispatch, seal, turn-end, and model-call phases
       |
       v
decision commit -> shared tool dispatch -> atomic bundle -> turn end -> model
       |
       v
ExecutionJournal -> task/replay/verification projections
```

The durable guarantee is deliberately narrow. Controlled `edit` operations
persist an intent and reconcile the actual file against before/after hashes.
The runtime does not claim general exactly-once execution for shell commands,
network APIs, or arbitrary external systems.

## Evidence ledger

| Contract | Implementation | Current evidence | Status |
| --- | --- | --- | --- |
| Decision is durable before tool effects | `commit_model_decision()` atomically writes the AssistantMessage, immutable batch plan, steps, and cursor | Public subprocess crash A | Demonstrated |
| Committed prefix is not repeated | Core derives progress from ordered result and execution-step facts | A/B/C assert 3/2/1 actual recovery writes respectively | Demonstrated for controlled edit |
| Applied edit without terminal result is reconciled | Edit before/after fingerprints and stored canonical ToolResult | Public three-call crash C plus patched write counter | Demonstrated for controlled edit |
| Bundle commit is atomic and batch-scoped | `commit_batch_bundle(batch_id, event)` writes message, commit fact, and cursor in one transaction | Public three-call crash D | Demonstrated |
| Committed bundle is not duplicated | `ToolBatchBundleCommitted` is keyed by batch identity | Public three-call crash E | Demonstrated |
| Recovery and normal execution share one driver | `core.run()` sends both new and recovered calls through `dispatch_tool_calls()`; no standalone recovery executor remains | Recovery emits the normal ToolExecution start/end stream | Demonstrated |
| Cancellation precedes recovery effects | Core checks `abort` before reconciliation and each unsettled durable call | Public A recovery tests cover entry cancellation and cancellation before the first tool | Demonstrated |
| Incomplete legacy history fails before effects | Pending canonical message validation runs under the writer lock before reconciliation | Public migration test F with retryable edit intent | Demonstrated |
| Recovery can itself crash and resume | A second process crashes after A result; the next process executes B/C only | Public subprocess test; all step attempts remain one | Demonstrated for controlled edit |
| Workspace drift blocks recovery | A controlled edit accepts only the recorded before/after fingerprint | Public C test changes the affected file to a third value and observes zero new writes | Demonstrated for controlled edit |
| Provider history preserves pairing | Canonical AssistantMessage and one ordered ToolResult bundle are rebuilt | A–E inspect actual `LLMRequest.messages` for IDs A/B/C | Demonstrated |
| ToolBatch is immutable plan state | Table contains only `batch_id`, `turn_no`, and `calls_json` | Schema test; progress derives from facts | Demonstrated for new schema |
| Historical sealed batch cannot authorize current recovery | `assert_resumable()` accepts only a current pending batch or current `seal_pending` cursor | Regression test with an old sealed batch plus a new uncommitted turn | Demonstrated |
| Acceptance contract is frozen | Contract ID, version, and commands are persisted when the task is created | Replacement-contract rejection test | Demonstrated |
| Receipt binds current task and code | Receipt carries task revision, tree hash, contract, and verifier identity | State-change and tree-change invalidation tests | Demonstrated |
| Completed status cannot bypass evidence currency | Re-entering the completion gate rechecks the accepted receipt against the current tree | Complete-then-change-code rejection test | Demonstrated |
| Failure classes are explicit | Receipts distinguish passed, code, verifier, infrastructure, and budget outcomes | Real child processes cover missing command, signal termination, timeout, and ordinary nonzero exit | Demonstrated locally |
| Fork is an independently resumable child task | `fork_task()` creates a child record, workspace, history, checkpoint, and lineage | Child resumes independently; parent Journal and files remain unchanged | Demonstrated |
| Context manifest and complete-boundary lineage | Every checkpoint writes a versioned context manifest; forks persist parent identity | Fork test checks boundary flag, message refs, snapshot/tree identity, lineage, resumability, and parent immutability | Demonstrated |
| Run diff supports controlled comparison | `diff_runs()` compares committed replay projections | Replay test covers status, tree, event delta, and history equality | Demonstrated |
| Sealed terminating batch does not resume the model | Core reconstructs `terminate` from durable results while closing `seal_pending` | E-window regression test observes a terminating TurnEnd and zero model calls | Demonstrated |
| Malformed batch plan fails closed | Journal rejects invalid call entries instead of filtering them | Corrupted SQLite plan regression test | Demonstrated |
| Regression runner is a formal runtime benchmark | Runner validates deterministic fixtures only | Preflight tests | Explicitly not claimed |

## A–F crash matrix

Every current A–E process test starts from the same model decision:
`Assistant [A, B, C]`. The next real model request must contain exactly one
Assistant tool-call message and one ToolResult bundle, both ordered A/B/C.

| Case | Crash point | Required observation |
| --- | --- | --- |
| A | Decision commit, before A | A/B/C execute in order without regenerating the decision |
| B | A result commit | A is skipped; B/C execute in order |
| C | B file effect, before B terminal result | B is reconciled and not written again; C executes |
| D | All results, before bundle transaction | No orphan bundle message exists before resume |
| E | Atomic bundle commit, before next ModelCall | Bundle remains single and the next request is valid |
| F | Legacy pending batch lacks AssistantMessage | Resume fails before edit reconciliation or model/tool execution |

The separate repeated-recovery test crashes after the original A boundary and
again after A result during recovery. A final resume uses the original decision
and completes only the remaining suffix.

## Release boundary

- Durable side-effect reconciliation is claimed only for the controlled
  `edit` tool. Shell commands, network APIs, and arbitrary external systems are
  outside the guarantee.
- Hook callbacks are lifecycle extensions, not durable side-effect handlers.
  Their external effects are not exactly-once across process death.
- `diff_runs()` is a structural comparison of committed facts, not a causal
  benchmark or a claim that one model/runtime configuration is better.
- The deterministic regression runner remains a preflight fixture runner, not
  a formal Runtime or model benchmark.

Memory expansion and additional experiments are outside this closeout scope.

## Current verification state

Focused command:

```bash
uv run python -m unittest \
  tests.unit.test_core \
  tests.unit.test_hooks \
  tests.unit.test_workspace \
  tests.unit.test_runtime_journal \
  tests.unit.test_tool_recovery \
  tests.unit.test_runtime_crash_contract \
  tests.unit.test_runtime_migration \
  tests.unit.test_evidence_completion \
  tests.unit.test_runtime_replay \
  tests.unit.test_durable_runtime_demo -v
```

Result: the focused core, workspace, recovery, migration, verification, replay,
fork, hook, and demo suite passed 133 tests after the current changes.

The key-free demo is:

```bash
bash runs/demos/run_durable_runtime_demo.sh
```

Its latest run injected exit code 91 after the decision commit, recovered A/B/C,
made one next model call, committed one bundle, passed a verifier that reads all
three output files, and reached `completed`. A negative test changes one output
and proves that the same contract then fails.

The latest broad unit run was:

```bash
uv run python -m unittest discover -s tests/unit
```

Result: 825 tests ran, 6 skipped, and all passed. The WSL-native full gate
`bash runs/dev/run_ci.sh` also passed format, lint, docs, architecture,
environment, type checks, unit tests, and the bash demo.

## Historical record

Earlier development added SQLite WAL journaling, canonical message and
ToolResult codecs, execution-step facts, controlled edit reconciliation,
snapshots, replay, verification receipts, explicit legacy import, and a
deterministic regression preflight. Historical full gates were reported green
at 798–806 tests depending on the commit.

Those historical runs remain useful evidence for their commits. They do not
prove the current worktree, and they never proved the complete recovery,
verification, fork, or run-diff contract. In particular, an earlier unit test
constructed out-of-order result rows and then sorted them during bundle
rebuild; the current runtime instead rejects out-of-order durable result facts.

The first full gate in the clean release worktree failed two tests because they
read ignored files under `evals/out/`. Those tests now build minimal temporary
selection, task, verifier, and report fixtures; the subsequent 825-test gate is
the reproducible result reported above.
