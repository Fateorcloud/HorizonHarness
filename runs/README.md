# Runs

Launch surface for the project: one Python entry over every benchmark, plus
small copy-pasteable shell scripts (nanochat's `runs/` convention — readable and
useful as the reference way to run an experiment), grouped by concern.

## Layout

```
runs/
  run_bench.py               # one entry (list/setup/<bench>/batch/score/oracle/all)
  bench-manifest.example.json
  profiles/                  # per-bench run-profiles (*.example.json)
  _benches/                  # internal per-bench modules (imported by run_bench.py)
  lib/                       # small shared launch helpers
  harbor/                    # Terminal-Bench 2.1 Harbor batch launchers
  swebench/                  # SWE-bench batch/setup/eval scripts
  programbench/              # ProgramBench batch/setup scripts
  demos/                     # bash-agent / MCP / trace-viewer demos
  dev/                       # run_ci.sh, run_docs_lint.sh
  task_workspace.py          # durable safe-turn task continuation
  memory_validation.py       # indexed keyword/hybrid validation
  workspace_memory_eval.py   # frozen S/W/K/Y matrix + budget + verifier
```

Start with the unified entry:

```bash
uv run python -m runs.run_bench list            # what benches exist
uv run python -m runs.run_bench setup           # is my environment ready?
uv run python -m runs.run_bench <bench> ...     # run one bench
uv run python -m runs.run_bench batch <bench> ...   # select/run a concurrent batch
uv run python -m runs.run_bench score <bench> ...   # official scorer (or note inline scoring)
uv run python -m runs.run_bench oracle <bench> ...  # gold/model-free reference run (wiring check)
uv run python -m runs.run_bench all --manifest runs/bench-manifest.json
```

`score` reaches a bench's official scorer (SWE-bench / ProgramBench delegate to
their `evals/<suite>/evaluate_*.py`; a bench that grades inline says so and does
nothing). `oracle` is sugar for the run path with `--provider oracle`, applying
the reference solution model-free as a deterministic wiring check (only benches
whose `--provider` accepts `oracle`).

## Available shell runs

```bash
bash runs/dev/run_ci.sh
bash runs/dev/run_docs_lint.sh
bash runs/demos/run_bash_agent_demo.sh
bash runs/harbor/run_terminal_bench_2_1_sequential.sh
bash runs/swebench/run_swebench.sh
bash runs/swebench/run_swebench.sh --variant multilingual
bash runs/swebench/run_swebench.sh --variant pro
bash runs/swebench/eval_swebench.sh
```

`runs/dev/run_ci.sh` mirrors the GitHub Actions workflow at
`.github/workflows/ci.yml`: it syncs the dev dependency group, checks Ruff
formatting and lint, checks docs plus generated references, enforces
architecture and environment-variable boundaries, runs `ty` on `src/`, runs
the full unittest suite, and runs the deterministic bash-agent demo smoke. Use
it as the local pre-push gate.

The Terminal-Bench 2.1 Harbor launchers run `bash` and `bash_task` experiments
with separate job names. The sequential entry waits for the complete blocking
Harbor process for `bash` before starting `bash_task`:

```bash
bash runs/harbor/run_terminal_bench_2_1_bash.sh
bash runs/harbor/run_terminal_bench_2_1_bash_task.sh
bash runs/harbor/run_terminal_bench_2_1_sequential.sh
```

Use `HARBOR_DRY_RUN=1` with any launcher to validate its command without
starting an experiment.

The focused tests cover the canonical runtime:

```bash
uv run python -m unittest discover -s tests/unit
```

## Task workspace and indexed memory

The key-free Workspace smoke uses separate CLI invocations, so resume cannot
reuse the first process's `State`:

```bash
uv run python -m runs.task_workspace --fake --store-dir evals/out/workspace-smoke \
  start "Return a short answer." --task-id smoke --pause-after-turns 1
uv run python -m runs.task_workspace --store-dir evals/out/workspace-smoke inspect smoke
uv run python -m runs.task_workspace --fake --store-dir evals/out/workspace-smoke \
  resume smoke --pause-after-turns 1
```

Omit `--fake` for the configured provider. The CLI supports only a single text
workflow and complete-turn checkpoints; it refuses an interrupted turn rather
than replaying possibly completed tool side effects. See
[`docs/workspace.md`](../docs/workspace.md).

`runs.memory_validation prepare` loads explicit candidates for storage/retrieval
fixtures only; it is not task evidence extraction. `run` records keyword or
real local-embedding retrieval; `report` combines the raw JSON outputs. Real
terminal-task extraction is wired through `runs.task_workspace
--extract-memory-on-complete`. Hybrid mode requires `uv sync --extra memory`
and an exact model revision. It does not silently claim a keyword fallback as
Hybrid.

The formal entry freezes the four selected tasks, implementation fingerprint,
fixed suffix order, three phase budgets and separate task/verifier hashes
without making a model call:

```bash
uv run --no-sync python -m runs.workspace_memory_eval plan
```

`run` is the explicit paid path. It uses `.env.history-reuse.glm` by default,
disables implicit model retries, reserves budget before every Agent, summary
and memory-extraction call, executes at most 3 D attempts plus the three shared
prefixes and fixed 10 suffixes, and invokes the official SWE-bench verifier
after each completed Agent path. Verifier JSONL never enters an Agent checkout
or command container:

```bash
uv run --no-sync python -m runs.workspace_memory_eval run \
  --development-attempts 1 \
  --embedding-revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42
```

The plan command is safe for offline validation. The run command spends real
model budget and should only be started after engineering checks pass.

The current authorized cumulative budget is 5,000,000 tokens; previous charges
remain in the ledger. Old experiment manifests retain their original budget.
Recovery outputs record the changed implementation and are not original frozen
scores. `run --repair-h3` reclassifies existing official empty-patch reports as
failures (not infrastructure errors), then retries only the two H3 continuations
under separate cell IDs. It requires the same `--embedding-revision` argument;
the original results and checkpoints remain available. Corrected results are
written to `repaired-results.json` in the output directory.

`run --combined-h3` runs only one `C-H3` continuation from the saved H3 prefix
with the existing summary plus compact current state and on-demand Workspace
reading. It reuses the summary generation cost rather than charging a new model
call. Results are in `combined-results.json`; this one-case engineering check
does not establish a stable efficiency improvement over historical S/W runs.

This checks local Markdown links and backticked path references:

```bash
bash runs/dev/run_docs_lint.sh
```

This runs a deterministic mini-SWE-style bash-use agent demo:

```bash
bash runs/demos/run_bash_agent_demo.sh
```

To verify SWE-bench adapter tests specifically (already included in `run_ci.sh`):

```bash
uv run python -m unittest \
  tests.unit.test_swebench_patch_extract \
  tests.unit.test_swebench_harness \
  tests.unit.test_swebench_evaluate_predictions
```

## SWE-bench (Containerized)

One-time setup: install Docker, build SWE-bench images for an instance:

```bash
bash runs/swebench/setup_swebench_docker.sh sympy__sympy-23824
```

These run the containerized SWE-bench agent for one default instance, one named
instance, a selected subset, or the full dataset split. Subset and full-split
runs can limit Docker/model concurrency with `--parallel N`; subset files use
one instance id per line and may include `#` comments. The shell script is a
six-line compatibility wrapper over `run_bench.py batch swebench`:

```bash
bash runs/swebench/run_swebench.sh
bash runs/swebench/run_swebench.sh sympy__sympy-23824
bash runs/swebench/run_swebench.sh --ids-file evals/out/swebench/ids.txt --parallel 4
bash runs/swebench/run_swebench.sh --all --parallel 4

bash runs/swebench/run_swebench.sh --variant multilingual
bash runs/swebench/run_swebench.sh --variant multilingual --ids-file ids.txt --parallel 4
bash runs/swebench/run_swebench.sh --variant multilingual --all --parallel 4

bash runs/swebench/run_swebench.sh --variant pro
bash runs/swebench/run_swebench.sh --variant pro instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08
bash runs/swebench/run_swebench.sh --variant pro --ids-file ids.txt --parallel 4
bash runs/swebench/run_swebench.sh --variant pro --all --parallel 4

# The same Python entry, without the compatibility wrapper:
uv run --extra swebench python -m runs.run_bench batch swebench \
  --variant pro --all --parallel 4
```

SWE-bench Verified artifacts land under `evals/out/swebench/`; SWE-bench
Multilingual and Pro artifacts land under sibling roots
`evals/out/swebench_multilingual/` and `evals/out/swebench_pro/`. All roots use
the same flat layout: optional `instance_<id>.jsonl` setup inputs,
`wheelhouse/` for wheels, and `<run-id>/<instance-id>/` for per-instance
outputs. The Python batch entry loads rows with the `datasets` package from
`uv sync --extra swebench` (which reuses the HuggingFace cache).

This wrapper runs or normalizes official SWE-bench / SWE-bench Pro evaluation
results for an existing predictions file:

```bash
bash runs/swebench/eval_swebench.sh --run-official --predictions evals/out/swebench_predictions.jsonl
bash runs/swebench/eval_swebench.sh --multilingual --predictions evals/out/swebench_multilingual/swebench_multilingual_predictions.jsonl
bash runs/swebench/eval_swebench.sh --pro --predictions evals/out/swebench_pro/swebench_pro_predictions.jsonl --results-json evals/out/swebench_pro_eval/eval_results.json
```

See `evals/swebench/README.md` for detailed Docker setup, macOS arm64
workarounds, and troubleshooting.

## History-reuse pilot (separate from benchmark scores)

`runs.history_reuse` tests constructed, prefilled single-task continuation,
not autonomous long-horizon performance. Its cases and external assertions live
in `runs/history_reuse_cases.py`. Normal bash/read/edit tools remain available.
The file-contract case can reread its contract; the historical-receipt case
deliberately models a transient observation and is not a historical failure.

Use a dedicated gitignored dotenv containing `OPENAI_AUTH_TOKEN`,
`OPENAI_BASE_URL`, and `OPENAI_MODEL`. The runner reads only that file, not the
root dotenv or ambient provider credentials. Run the offline checks first:

```bash
uv run python -m unittest tests.unit.test_history_reuse -v
# Paid: one request, 64 output tokens maximum, no SDK retries.
uv run python -m runs.history_reuse --dotenv .env.history-reuse.glm \
  --output evals/out/history-reuse/preflight-new
# Paid: two constructed cases, each with A/B/C, one repetition.
uv run python -m runs.history_reuse --pilot --dotenv .env.history-reuse.glm \
  --output evals/out/history-reuse/pilot-new
```

Never reuse an output directory. The pilot writes manifests, initial workspace
snapshots, external grading output, patches, event traces, raw request/response
bodies and per-cell results. These artifacts are local; do not merge them into
formal benchmark aggregates. `passed` means external code assertions passed,
not that the model finished cleanly; inspect `status` and the agent-end event.

The pilot keeps AgentSession assembly and the ordinary loop, but replaces the
retrying model closure with a single-attempt SDK boundary using the existing
wire converters. This deliberate experimental difference is not a production
adapter change. The manifest records per-run limits; input admission uses a
conservative byte proxy rather than the provider's tokenizer, while reported
usage is counted after each call. Proxy limits may stop a run early. Unpriced
usage is reported with null cost, never zero. The historical seed commands are
constructed observations, not programs installed in the test workspace.

After the index intervention, `--groups D --pilot` runs only the two treatment
cells. A/B/C retain their original no-citation format through the experiment's
`history_reuse_policy.py`; do not compare a newly modified baseline to the old
scores. A single run per cell is exploratory, not a statistically established
improvement. Keep raw artifacts private: unrestricted bash output can contain
host environment information even though the runner never exports its API key.

`--split holdout --groups ABCD` selects the reserved retry-classification and
legacy-wire-parser tasks. They use different task logic and fact indices from
the debug cases; they are still constructed continuation tasks, not independent
real-world benchmarks. Freeze code and assertions before the first model run.
Once results are inspected, do not tune on them and still call them untouched
holdouts. Eight cells have a ceiling of 96 requests, 512000 input and 64000 output
tokens under the same per-cell guards; approve that batch budget separately.
