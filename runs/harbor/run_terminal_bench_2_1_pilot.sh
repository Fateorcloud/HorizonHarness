#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

pilot_file="runs/harbor/terminal_bench_2_1_pilot.txt"
if [[ ! -f "$pilot_file" ]]; then
  echo "Missing Pilot task list: $pilot_file" >&2
  exit 2
fi

dataset="${HARBOR_DATASET:-terminal-bench/terminal-bench-2-1}"

dataset_args=()
if [[ -d "$dataset" ]]; then
  dataset_args+=(--path "$dataset")
else
  dataset_args+=(--dataset "$dataset")
fi

default_timestamp="$(date +%Y%m%d-%H%M%S)"
run_timestamp="${RUN_TIMESTAMP:-$default_timestamp}"
job_name="${HARBOR_JOB_NAME:-horizon-tb21-pilot-$run_timestamp}"

task_args=()
while IFS= read -r task_name; do
  [[ -z "$task_name" || "$task_name" == \#* ]] && continue
  task_id="$task_name"
  if [[ ! -d "$dataset" ]]; then
    task_id="terminal-bench/$task_name"
  fi
  task_args+=(--include-task-name "$task_id")
done < "$pilot_file"

extra_args=()
if [[ "${HARBOR_DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

uv run python -m runs.run_bench harbor \
  --dotenv .env \
  "${dataset_args[@]}" \
  --job-name "$job_name" \
  --n-concurrent "${HARBOR_CONCURRENCY:-5}" \
  --max-turns 150 \
  --max-output-tokens 4096 \
  --timeout-multiplier 3 \
  --agent-setup-timeout-multiplier 3 \
  --agent-kwarg install_timeout_sec=900 \
  --agent-env HORIZON_BASH_MAX_TIMEOUT_SECONDS=300 \
  --setup-proxy-from-env \
  --capabilities bash,read,task \
  --workflow single \
  --api-kind openai-chat \
  --debug \
  --json \
  "${task_args[@]}" \
  "${extra_args[@]}"
