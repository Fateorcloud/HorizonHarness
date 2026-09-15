#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run --extra harbor python -m runs.harbor.run_terminal_bench_2_1_baseline_multikey "$@"
