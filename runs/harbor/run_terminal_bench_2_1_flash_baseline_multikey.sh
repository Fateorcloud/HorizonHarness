#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run --extra harbor python -m runs.harbor.run_terminal_bench_2_1_baseline_multikey \
  --model deepseek-ai/deepseek-v4-flash-0731 \
  --key-env OPENAI_AUTH_TOKEN_4 \
  --key-env OPENAI_AUTH_TOKEN_5 \
  --key-env OPENAI_AUTH_TOKEN_6 \
  "$@"
