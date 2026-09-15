#!/usr/bin/env python3
"""Run the fixed 18-task Terminus-2 Baseline across three API keys."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from horizon_harness.llm.env import load_dotenv
from runs.lib.harbor_shards import (
    BASELINE_MODEL,
    build_baseline_shard_command,
    merge_harbor_results,
    partition_tasks,
    read_task_names,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_FILE = ROOT / "runs/harbor/terminal_bench_2_1_fixed_18.txt"
DEFAULT_JOBS_DIR = ROOT / "evals/out/harbor/jobs"
DEFAULT_AGGREGATES_DIR = ROOT / "evals/out/harbor/aggregates"
DEFAULT_KEY_ENV_NAMES = (
    "OPENAI_AUTH_TOKEN",
    "OPENAI_AUTH_TOKEN_2",
    "OPENAI_AUTH_TOKEN_3",
)


@dataclass(frozen=True)
class Launch:
    slot: int
    key_env: str
    job_name: str
    tasks: tuple[str, ...]
    command: list[str]
    result_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dotenv", type=Path, default=ROOT / ".env")
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--aggregates-dir", type=Path, default=DEFAULT_AGGREGATES_DIR)
    parser.add_argument("--model", default=BASELINE_MODEL)
    parser.add_argument(
        "--key-env",
        action="append",
        dest="key_env_names",
        default=None,
        help=(
            "Environment variable holding one shard key. Repeat exactly three "
            "times; defaults to OPENAI_AUTH_TOKEN, _2, and _3."
        ),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _launches(args: argparse.Namespace) -> tuple[Launch, ...]:
    tasks = read_task_names(args.tasks_file)
    shards = partition_tasks(tasks, shard_count=3, tasks_per_shard=6)
    key_env_names = tuple(args.key_env_names or DEFAULT_KEY_ENV_NAMES)
    if len(key_env_names) != 3:
        raise ValueError("provide exactly three --key-env values")
    if len(set(key_env_names)) != 3:
        raise ValueError("the three --key-env values must be unique")
    model_name = args.model.removeprefix("openai/").rsplit("/", 1)[-1]
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-")
    run_name = args.run_name or (
        f"baseline-tb21-{model_slug}-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    launches: list[Launch] = []
    for slot, (key_env, shard) in enumerate(zip(key_env_names, shards), start=1):
        job_name = f"{run_name}-key{slot}"
        command = build_baseline_shard_command(
            python_executable=sys.executable,
            dotenv=args.dotenv.resolve(),
            jobs_dir=args.jobs_dir.resolve(),
            job_name=job_name,
            tasks=shard,
            model=args.model,
        )
        launches.append(
            Launch(
                slot=slot,
                key_env=key_env,
                job_name=job_name,
                tasks=shard,
                command=command,
                result_path=args.jobs_dir.resolve() / job_name / "result.json",
            )
        )
    return tuple(launches)


def _safe_launch_payload(
    run_name: str, model: str, launches: tuple[Launch, ...]
) -> dict[str, Any]:
    return {
        "schema": "horizon-harness.harbor-shard-launch.v1",
        "run_name": run_name,
        "baseline": "harbor-terminus-2",
        "model": model,
        "key_count": len(launches),
        "concurrency_per_key": 2,
        "total_concurrency": len(launches) * 2,
        "shards": [
            {
                "slot": launch.slot,
                "key_env": launch.key_env,
                "job_name": launch.job_name,
                "tasks": list(launch.tasks),
                "command": launch.command,
                "result_path": str(launch.result_path),
            }
            for launch in launches
        ],
    }


def run(args: argparse.Namespace) -> int:
    env = dict(os.environ)
    load_dotenv(args.dotenv, environ=env)
    launches = _launches(args)
    run_name = launches[0].job_name.rsplit("-key", 1)[0]
    launch_payload = _safe_launch_payload(run_name, args.model, launches)
    if args.dry_run:
        print(json.dumps(launch_payload, ensure_ascii=False, indent=2))
        return 0

    key_env_names = tuple(launch.key_env for launch in launches)
    missing = [name for name in key_env_names if not env.get(name, "").strip()]
    if missing:
        raise SystemExit(
            f"missing API key variables in {args.dotenv}: {', '.join(missing)}"
        )
    keys = [env[name].strip() for name in key_env_names]
    if len(set(keys)) != len(keys):
        raise SystemExit("the three API key variables must contain distinct key values")

    aggregate_dir = args.aggregates_dir.resolve() / run_name
    _write_json(aggregate_dir / "launch.json", launch_payload)
    processes: list[tuple[Launch, subprocess.Popen[str], TextIO]] = []
    for launch in launches:
        child_env = dict(env)
        key = env[launch.key_env].strip()
        child_env["OPENAI_AUTH_TOKEN"] = key
        child_env["OPENAI_API_KEY"] = key
        child_env["OPENAI_MODEL"] = args.model.removeprefix("openai/")
        log_file = (aggregate_dir / f"key{launch.slot}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            launch.command,
            cwd=ROOT,
            env=child_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((launch, process, log_file))
        print(
            f"started key{launch.slot}: {launch.job_name} "
            f"({len(launch.tasks)} tasks, concurrency 2)"
        )

    return_codes: dict[str, int] = {}
    for launch, process, log_file in processes:
        try:
            return_codes[launch.job_name] = process.wait()
        finally:
            log_file.close()
        print(f"finished key{launch.slot}: exit={return_codes[launch.job_name]}")

    missing_results = [
        str(launch.result_path)
        for launch in launches
        if not launch.result_path.is_file()
    ]
    final_launch = {
        **launch_payload,
        "return_codes": return_codes,
        "missing_results": missing_results,
    }
    _write_json(aggregate_dir / "launch.json", final_launch)
    if missing_results:
        print(
            "cannot merge; missing shard results:\n" + "\n".join(missing_results),
            file=sys.stderr,
        )
        return 1

    merged = merge_harbor_results(
        [launch.result_path for launch in launches], merged_job_name=run_name
    )
    merged["launch_manifest"] = str(aggregate_dir / "launch.json")
    result_path = aggregate_dir / "result.json"
    _write_json(result_path, merged)
    print(f"merged result: {result_path}")
    return 0 if all(code == 0 for code in return_codes.values()) else 1


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
