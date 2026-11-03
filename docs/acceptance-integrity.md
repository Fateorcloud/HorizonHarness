# Acceptance integrity

Task completion requires a successful receipt for the current task revision,
frozen contract, verifier version, and code tree. An exit code of zero alone
does not authorize completion.

## Frozen acceptance tests

An optional `AcceptanceContract.acceptance_dir` identifies host-managed tests.
It must exist, contain regular files, have no symlinks, and be disjoint from the
code directory: neither directory may contain the other. Creation stores an
unfiltered snapshot in `WorkspaceRecord.acceptance_snapshot`, its
`acceptance_tree_sha256`, and the original absolute `acceptance_dir`. File
contents live in the task's existing `snapshot-objects` directory.

The manifest, hash, directory, commands, identity/version, and verification
exclusions are immutable through `WorkspaceStore.save`. Verification and
completion reject a supplied contract that changes these frozen values. A
caller may omit the acceptance hash; `TaskSession` fills it from the task
record. An explicitly supplied hash must match. Path normalization is lexical,
so deleting or replacing the original directory after creation does not affect
verification.

Each verification materializes and validates the frozen snapshot in a fresh
temporary directory outside the code directory and task store. Commands receive
its absolute path in `HORIZON_ACCEPTANCE_DIR`; their working directory remains
the code directory. Temporary files are removed after both success and failure.
The original acceptance directory is never reread. Forks inherit the manifest,
hash, rules, and frozen content objects, and require independent verification.

A missing/corrupt object or materialization/hash failure produces
`acceptance_corrupted` without starting the command. The receipt retains the
expected acceptance hash. Completion rejects that receipt.

Commands must actually read the tests through `HORIZON_ACCEPTANCE_DIR`. This
mechanism freezes their input; commands and their dependencies remain trusted
host code, as in the existing local verifier contract.

## Verification tree and command effects

`verification_exclude_dirs` always includes `.git`, `__pycache__`,
`.pytest_cache`, `.mypy_cache`, and `.ruff_cache`. A contract may append plain
directory names; paths and `.`/`..` are rejected. Rules match directory path
components at any depth. An ordinary file named `.git` or `.pytest_cache.txt`
remains part of the hash. Effective rules are frozen and written to receipts.

Only verification and receipt freshness use these rules. Checkpoints, drift
comparison, restore, and fork materialization retain the existing unfiltered
`capture_tree` hash, including caches and Git data. A legacy receipt with no
exclusion field uses the original unfiltered hash when checking freshness.
Legacy task records and contracts without `acceptance_dir` require no
acceptance snapshot. New receipts include the effective verification rules;
completion for a newly frozen contract requires those same rules.

The verifier hashes code before and after command execution. Receipts bind the
before hash. A changed nonexcluded tree yields
`verifier_modified_workspace`, even with exit code zero, and lists sorted
`workspace_changes` as `added:path`, `changed:path`, or `removed:path`. Results
and changes survive reopening the task. The runtime leaves all command effects
in place; restoring the files manually does not turn a failed receipt into a
successful one.

Outcome priority is:

1. `acceptance_corrupted` (commands do not run).
2. `budget_stopped`, `infrastructure_failed`, or `verifier_failed` (retain the
   process failure classification, and also record observed workspace changes).
3. `verifier_modified_workspace`.
4. `code_failed` or `passed`.

## CLI

Freeze external tests when starting a task:

```bash
uv run --with pytest python -m runs.task_workspace --fake \
  --code-dir /tmp/example/code --store-dir /tmp/example/tasks \
  start 'Implement the answer' --task-id answer --pause-after-turns 1 \
  --acceptance-dir /tmp/example/acceptance \
  --verification-exclude-dir build \
  --check 'python -m pytest "$HORIZON_ACCEPTANCE_DIR"'
```

`--acceptance-dir` and `--verification-exclude-dir` require `--check`.
`verify` and `complete` load the frozen directory, hash, and exclusions from
the task record. Existing `--check` and `--contract-id` arguments still have to
match the frozen commands and identity:

```bash
uv run --with pytest python -m runs.task_workspace --fake \
  --code-dir /tmp/example/code --store-dir /tmp/example/tasks \
  verify answer --check 'python -m pytest "$HORIZON_ACCEPTANCE_DIR"'
uv run --with pytest python -m runs.task_workspace --fake \
  --code-dir /tmp/example/code --store-dir /tmp/example/tasks \
  complete answer --check 'python -m pytest "$HORIZON_ACCEPTANCE_DIR"'
```

The deterministic regression suite includes real pytest execution, tampered
acceptance sources and objects, temporary-directory cleanup, failure priority,
legacy receipts, forks, CLI round trips, and a fixed checkpoint hash:

```bash
uv run --with pytest python -m pytest tests/unit/test_acceptance_integrity.py -p no:cacheprovider
```
