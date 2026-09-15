#!/usr/bin/env bash
set -euo pipefail

uv run python -m runs.demos.durable_runtime_demo "$@"
