# Configuration Reference: Environment Variables

Every environment variable the project reads, grouped by the layer that owns it.
A *discoverability* surface — code is still the source of truth for behavior.
Each variable's name constant is declared in exactly one module (the "Owner"
column); other call sites import it rather than re-declaring the string.

For launching runs (`runs/run_bench.py`, `--profile` JSON bundles), see
[`runs/README.md`](../runs/README.md).

## Declaring a new knob

Declare it as an `EnvVar` in `src/horizon_harness/config.py` — a leaf module
that declares names and imports nothing internal, so layers depend on it and
never the reverse. `scripts/env_lint.py` enforces this in CI: a direct
`os.environ` read under `src/horizon_harness/` outside an owner module below
fails the build unless the line carries `# env-ok: <reason>`.

Knobs not yet migrated stay with the layer that owns the concern (see the
Owner lines below), in a *light* module so host harnesses can forward names
into a container without importing a heavy graph. Do **not** centralize those
into one module if it would couple unrelated layers.

## Registry-backed configuration

Generated from `REGISTRY` and validated in CI, so it cannot drift.

<!-- BEGIN GENERATED: config-registry (scripts/build_config_reference.py) -->
<!-- Generated from horizon_harness.config.REGISTRY — do not edit by hand; run scripts/build_config_reference.py. -->

### `agent.compression`

| Variable | Default | Purpose |
| --- | --- | --- |
| `HORIZON_AGENT_COMPRESSION_THRESHOLD_TOKENS` | unset | Token threshold that triggers compression; default is window * ratio, else a fixed fallback. |
| `HORIZON_AGENT_COMPRESSION_WINDOW_RATIO` | `0.8` | Fraction of the context window used as the threshold when none is set. |
| `HORIZON_AGENT_COMPRESSION_KEEP_RECENT` | `4` | Recent turns kept verbatim during compression. |

### `agent.llm`

| Variable | Default | Purpose |
| --- | --- | --- |
| `HORIZON_LLM_REQUEST_TIMEOUT_SECONDS` | unset | Per-request model API timeout in seconds; unset uses the adapter default. |

### `agent.tools`

| Variable | Default | Purpose |
| --- | --- | --- |
| `HORIZON_BASH_DEFAULT_TIMEOUT_SECONDS` | unset | Default timeout for bash tool commands; unset uses the tool default. |
| `HORIZON_BASH_MAX_TIMEOUT_SECONDS` | unset | Maximum model-selectable timeout for bash tool commands; unset uses the tool default. |
| `HORIZON_BASH_MAX_OUTPUT_CHARS` | unset | Maximum model-visible characters per bash output stream; unset uses the tool default. |
| `HORIZON_BASH_SUBMISSION_MARKER` | `` | When set, a bash command whose first output line matches this marker terminates the run and stores the remaining output as a submission. |

### `agent.workflow`

| Variable | Default | Purpose |
| --- | --- | --- |
| `HORIZON_WORKFLOW_WORKER_MAX_TURNS` | `40` | Per-worker inner turn budget. |
| `HORIZON_WORKFLOW_LOOP_MAX_TURNS` | `6` | Loop workflow: judge-gated outer iterations. |
| `HORIZON_WORKFLOW_PDR_ROUNDS` | `2` | PDR workflow: distill/refine rounds. |
| `HORIZON_WORKFLOW_PDR_WIDTH` | `3` | PDR workflow: parallel attempts per round. |
| `HORIZON_WORKFLOW_PDR_ATTEMPT_TURNS` | unset | PDR workflow: per-attempt turn budget; defaults to HORIZON_WORKFLOW_WORKER_MAX_TURNS. |

### `eval.onemillion`

| Variable | Default | Purpose |
| --- | --- | --- |
| `HORIZON_OMB_REFLECTION_ROUNDS` | `2` | Reflection rounds. |
| `HORIZON_OMB_PARALLEL_WORKERS` | `3` | Parallel workers. |
| `HORIZON_OMB_PDR_ROUNDS` | `2` | PDR rounds. |
| `HORIZON_OMB_PDR_WIDTH` | `3` | PDR width. |
| `HORIZON_OMB_TIMEOUT` | `600.0` | Per-request timeout for every sub-agent (seconds). |

### `eval.swebench`

| Variable | Default | Purpose |
| --- | --- | --- |
| `HORIZON_SWE_REPO_LANGUAGE` | `python` | Repo language hint for the SWE-bench container. |

### `trace`

| Variable | Default | Purpose |
| --- | --- | --- |
| `HORIZON_LIVE_TRACE_PATH` | unset | Bind-mounted path for incremental live trace output (unset = off). |

<!-- END GENERATED: config-registry -->

## Provider / credentials

Owner: `src/horizon_harness/llm/env.py` (single source of truth).

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_MODEL` | required | Model id under test. |
| `OPENAI_AUTH_TOKEN` | required | Gateway bearer token (NOT `OPENAI_API_KEY`). |
| `OPENAI_BASE_URL` | unset | Endpoint override; must agree with `HORIZON_API_KIND`. |
| `HORIZON_API_KIND` | `openai-chat` | Wire adapter: `openai-chat`, `openai-responses`, `anthropic-messages`. |
| `OPENAI_SESSION_ID` | unset | Optional gateway trace header. |
| `OPENAI_LOG_ID` | unset | Optional gateway trace header (`X-TT-logid`). |
| `HORIZON_REASONING_EFFORT` | unset | Provider-agnostic reasoning depth. |
| `HORIZON_JUDGE_MODEL` | falls back to `OPENAI_MODEL` | Eval-grader model. |
| `HORIZON_JUDGE_AUTH_TOKEN` | falls back to `OPENAI_AUTH_TOKEN` | Eval-grader token. |
| `HORIZON_JUDGE_BASE_URL` | falls back to `OPENAI_BASE_URL` | Eval-grader endpoint. |
| `HORIZON_JUDGE_API_KIND` | falls back to `HORIZON_API_KIND` | Eval-grader wire adapter. |

The eval harnesses accept only `openai-chat` / `openai-responses` via their own
`API_KIND_CHOICES` — an intentionally narrower set than `llm.env`'s.

## Everything else

| Variable | Owner | Default | Purpose |
| --- | --- | --- | --- |
| `MODEL_CONFIG` | `horizon_harness.llm.config` | unset | Path to a `models.json` mapping aliases (`strong`/`fast`) to provider specs. |
| `MODEL_CONFIG_JSON` | `horizon_harness.llm.config` | unset | Same schema inline, for sandboxes where writing a file is awkward. |
| `<ALIAS>_*` | `horizon_harness.llm.config` | falls back to `OPENAI_*` | Per-alias provider env (e.g. `STRONG_MODEL`); single-model setups collapse onto `OPENAI_*`. |
| `HORIZON_PRICE_BOOK` | `horizon_harness.model_metadata` | built-in | JSON model→rate override, merged over the built-in price book. |
| `HORIZON_CONTEXT_WINDOW_BOOK` | `horizon_harness.model_metadata` | built-in | Model metadata with context windows (LiteLLM / models.dev formats). |
| `HORIZON_CAPABILITIES` | `horizon_harness.spec` | `bash,read,task` | Comma-separated capabilities. |
| `HORIZON_WORKFLOW` | `horizon_harness.spec` | `single` | Workflow recipe name. |
| `PROGRAMBENCH_REQUIRE_NET_ISOLATION` | programbench suite | on | Require per-command network isolation (`unshare --net`). |
| `HORIZON_MEMORY_HOME` | `horizon_harness.evals.protocols` | unset (off) | Filesystem memory root; presence opts the run into memory. See `docs/memory.md`. |
| `HORIZON_MEMORY_NAME` | `horizon_harness.evals.protocols` | unset | Memory namespace. |
| `HORIZON_MEMORY_RUN_ID` | `horizon_harness.evals.protocols` | unset | Run id scoping memory artifacts. |
| `E2E_TRACE_PATH` | e2e tests | unset | Live e2e test trace output path (test-only, not in the registry). |
