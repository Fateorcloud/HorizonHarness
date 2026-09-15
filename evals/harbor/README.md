# Harbor Eval Integration

The `harbor` bench delegates dataset resolution, task environments, verifiers,
artifact collection, and aggregate scores to Harbor. HorizonHarness supplies the
installed agent and runs the standard `AgentSession` loop inside each task
environment.

The Harbor entry point is:

```text
horizon_harness.evals.harbor.agent:HorizonHarnessHarborAgent
```

The adapter installs the local `horizon-harness` source or wheel in the task
environment, then starts `horizon_harness.evals.harbor.runner`. The runner uses
the same capability/workflow contract as the rest of the project:
`capabilities=bash,read,task` and `workflow=single` for the standard
Terminal-Bench path. There is no host-side command forwarding tool.

## Boundary

Harbor owns:

- dataset and task resolution (`--dataset`, `--path`, `--repo`, `--task`);
- task lifecycle and resource/network policy;
- verifier execution and artifact download;
- job-level `result.json` aggregation.

HorizonHarness owns:

- the installed-agent adapter and container-local runner;
- the Agent Loop and bash/read/task tools;
- HorizonHarness summary and trajectory artifacts.

The `--agent-env KEY=${KEY}` entries shown by `--dry-run` are templates. Harbor
resolves them from the host environment for a real run, so dry-run output does
not expose credentials.

## Install and Dry Run

Harbor is optional:

```bash
uv sync --extra harbor
```

Inspect the generated command without Harbor, Docker, or model credentials:

```bash
uv run python -m runs.run_bench harbor \
  --dataset demo --n-tasks 1 --dry-run --json
```

The command contains the `HorizonHarnessHarborAgent` import path above and
passes `--agent-kwarg` values such as `capabilities=bash,read,task`,
`workflow=single`, and `max_turns=150`.

## Real Run

Use one root `.env` for all hosted-model runs. Copy `.env.example` to `.env`,
fill in `OPENAI_AUTH_TOKEN`, and change only `OPENAI_MODEL` when switching
models. For NVIDIA API Catalog hosted NIM, use:

```env
OPENAI_MODEL=<NVIDIA API Catalog model id>
OPENAI_AUTH_TOKEN=<NVIDIA API key>
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
HORIZON_API_KIND=openai-chat
```

```bash
uv run python -m runs.run_bench harbor \
  --dotenv .env --dataset <dataset-name> --n-tasks 1
```

For a local dataset or task directory, use `--path /path/to/input` instead.
Selectors are forwarded directly to Harbor:

```text
--dataset, --path, --repo, --task, --registry-url, --registry-path,
--include-task-name, --exclude-task-name, --n-tasks
```

HorizonHarness runner controls are passed as agent kwargs:

```text
--capabilities bash,read,task
--workflow single
--max-turns 150
--provider openai|fake
--api-kind openai-responses|openai-chat|anthropic-messages
--agent-kwarg install_timeout_sec=3000
--setup-proxy-from-env
```

Provider credentials use `OPENAI_MODEL`, `OPENAI_AUTH_TOKEN`, and the canonical
`HORIZON_API_KIND` / `HORIZON_REASONING_EFFORT` environment variables. The
setup-only proxy entries use the private `HORIZON_HARBOR_SETUP_*` prefix and are
filtered before the agent runner starts.

## Outputs

Harbor jobs default to `evals/out/harbor/jobs/`. Each trial's agent logs include:

```text
horizon-instruction.txt
horizon-summary.json
horizon-trajectory.jsonl
horizon-harness.txt
```

Harbor's `result.json` is the score source of truth; the HorizonHarness summary
and trajectory are debugging and trace artifacts.

Runs launched through `runs.run_bench` also write a secret-free
`horizon-harness.run-manifest.v1` sidecar before Harbor starts. After Harbor
creates the job directory, the same frozen launch record is copied to
`<job-dir>/run-manifest.json`; the JSON command result reports its path.

## Terminal-Bench 2.1

The fixed 20-task Pilot is an unbiased, reproducible sample of the 89 official
task names: sort by `SHA-256(task_name)`, take the first 20, then store the
reviewable list in `runs/harbor/terminal_bench_2_1_pilot.txt`. Run it with:

```bash
bash runs/harbor/run_terminal_bench_2_1_pilot.sh
```

The script defaults to the root `.env`, `single + bash/read/task`, five
concurrent tasks, and `max_turns=150`. Set `HARBOR_DRY_RUN=1` to inspect the command and
Run Manifest without starting containers.

### Three-key Terminus-2 Baseline

The quick Baseline comparison uses Harbor's built-in `terminus-2` agent on the
same fixed 18 tasks already completed by the DeepSeek HorizonHarness pilot. The
18-task list is the Pilot list without `code-from-image` and `gcode-to-text`,
matching the earlier Run Manifest exactly. This is an **initial reference
comparison**, not a single-variable Harness ablation: Terminus-2 has its own
terminal protocol and system prompt.

Keep all three credentials in the root `.env`:

```dotenv
OPENAI_MODEL=deepseek-ai/deepseek-v4-pro-0813
OPENAI_AUTH_TOKEN=first-key
OPENAI_AUTH_TOKEN_2=second-key
OPENAI_AUTH_TOKEN_3=third-key
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
```

The V4 Flash comparison uses three additional variables, leaving the V4 Pro
group unchanged:

```dotenv
OPENAI_AUTH_TOKEN_4=fourth-key
OPENAI_AUTH_TOKEN_5=fifth-key
OPENAI_AUTH_TOKEN_6=sixth-key
```

Run a secret-free launch preview, then start all three shards:

```bash
bash runs/harbor/run_terminal_bench_2_1_baseline_multikey.sh --dry-run
bash runs/harbor/run_terminal_bench_2_1_baseline_multikey.sh
```

The first command pair defaults to
`deepseek-ai/deepseek-v4-pro-0813` and key slots 1--3. The matching V4 Flash
reference run uses the same 18 tasks and budgets with key slots 4--6:

```bash
bash runs/harbor/run_terminal_bench_2_1_flash_baseline_multikey.sh --dry-run
bash runs/harbor/run_terminal_bench_2_1_flash_baseline_multikey.sh
```

The launcher deterministically assigns six tasks to each key and starts three
independent Harbor jobs with `--n-concurrent 2`, for at most six simultaneous
trials. A key never rotates inside a task. It passes the selected credential to
its child process through the environment, bridges it to LiteLLM's
`OPENAI_API_KEY` name in memory, and never writes the value to a command, Run
Manifest, log, or aggregate.

Each Harbor job remains under `evals/out/harbor/jobs/`. After all three finish,
the combined score and source-result paths are written to:

```text
evals/out/harbor/aggregates/<run-name>/result.json
```

The adjacent `launch.json` records the task-to-key-slot mapping and commands;
`key1.log`, `key2.log`, and `key3.log` retain per-shard console output. If a
shard does not produce `result.json`, the launcher exits non-zero and leaves the
other artifacts intact for diagnosis.

The checked-in launchers use the root `.env`, default
`HORIZON_REASONING_EFFORT` to `xhigh`, run ten trials concurrently, and set the
standard `single + bash/read/task` configuration with `max_turns=150`:

```bash
HARBOR_DRY_RUN=1 bash runs/harbor/run_terminal_bench_2_1_bash.sh
HARBOR_DRY_RUN=1 bash runs/harbor/run_terminal_bench_2_1_bash_task.sh
```

The two launchers are retained as separate historical experiment names but now
use the same explicit capability/workflow axes. The sequential wrapper waits
for the first Harbor process before starting the second:

```bash
bash runs/harbor/run_terminal_bench_2_1_sequential.sh
```

These commands are operator-run benchmark entry points, not part of CI and do
not run automatically during installation.
