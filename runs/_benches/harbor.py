"""Run Harbor datasets with HorizonHarness as a Harbor installed agent.

This is a thin host wrapper over ``harbor run``. Harbor owns dataset/task
resolution, environment lifecycle, verification, artifacts, and aggregate
results. HorizonHarness supplies the installed-agent import path that Harbor
installs and starts inside each task environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from horizon_harness.evals.harbor import (
    AGENT_IMPORT_PATH,
    DEFAULT_API_KIND,
    DEFAULT_MAX_TURNS,
)
from horizon_harness.evals.harbor.results import (
    find_latest_job_dir,
    summarize_result_file,
)
from horizon_harness.llm.env import load_dotenv
from horizon_harness.llm.env import (
    API_KIND_ENV,
    OPENAI_LOG_ID_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_SESSION_ID_ENV,
    REASONING_EFFORT_ENV,
)
from horizon_harness.spec import CAPABILITIES, DEFAULT_CAPABILITIES, WORKFLOWS

ROOT = Path(__file__).resolve().parents[2]
NAME = "harbor"
DESCRIPTION = (
    "Harbor datasets through Harbor's container/verifier harness with HorizonHarness as "
    "an installed agent."
)

DEFAULT_JOBS_DIR = ROOT / "evals/out/harbor/jobs"
HORIZON_AGENT = "horizon-harness"
TERMINUS_BASELINE_AGENT = "terminus-2"
_AGENT_LOG_INCLUDES = (
    "horizon-*.json",
    "horizon-*.jsonl",
    "horizon-harness.txt",
)
_PASS_ENV_KEYS = (
    OPENAI_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    API_KIND_ENV,
    REASONING_EFFORT_ENV,
    OPENAI_SESSION_ID_ENV,
    OPENAI_LOG_ID_ENV,
)
_SETUP_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
_SETUP_ENV_PREFIX = "HORIZON_HARBOR_SETUP_"
_DEFAULT_SETUP_ENV = {
    "PIP_INDEX_URL": "https://pypi.org/simple",
}
_SENSITIVE_ENV_RE = re.compile(
    r"(KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Path to a JSON run-profile (its `env` fills env gaps, its `run` "
            "flags are defaults overridable by explicit flags)."
        ),
    )
    parser.add_argument(
        "--harbor-bin",
        default="harbor",
        help="Harbor executable to run (default: harbor).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print/return the Harbor command without executing it.",
    )
    parser.add_argument(
        "--dotenv",
        default=str(ROOT / ".env"),
        help="Load env vars from this file before forwarding agent env.",
    )
    parser.add_argument(
        "--harbor-agent",
        choices=(HORIZON_AGENT, TERMINUS_BASELINE_AGENT),
        default=HORIZON_AGENT,
        help=(
            "Agent implementation to benchmark. Use terminus-2 for the Harbor "
            "reference Baseline; the default is HorizonHarness."
        ),
    )

    # Harbor dataset/task selectors. SAL forwards these instead of interpreting
    # Harbor datasets itself.
    parser.add_argument("-p", "--path", default=None)
    parser.add_argument("-d", "--dataset", default=None)
    parser.add_argument("--repo", default=None)
    parser.add_argument("-t", "--task", default=None)
    parser.add_argument("--registry-url", default=None)
    parser.add_argument("--registry-path", default=None)
    parser.add_argument(
        "-i",
        "--include-task-name",
        action="append",
        default=[],
        dest="include_task_names",
    )
    parser.add_argument(
        "-x",
        "--exclude-task-name",
        action="append",
        default=[],
        dest="exclude_task_names",
    )
    parser.add_argument("-l", "--n-tasks", type=int, default=None)

    # Harbor job controls.
    parser.add_argument(
        "--job-name",
        default=f"horizon-harbor-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument("--jobs-dir", default=str(DEFAULT_JOBS_DIR))
    parser.add_argument("-n", "--n-concurrent", type=int, default=None)
    parser.add_argument("-k", "--n-attempts", type=int, default=None)
    parser.add_argument("-r", "--max-retries", type=int, default=None)
    parser.add_argument("--timeout-multiplier", type=float, default=None)
    parser.add_argument("--agent-timeout-multiplier", type=float, default=None)
    parser.add_argument("--agent-setup-timeout-multiplier", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--yes", dest="yes", action="store_true", default=True)
    parser.add_argument("--no-yes", dest="yes", action="store_false")

    # Harbor environment controls.
    parser.add_argument("-e", "--env", default=None)
    parser.add_argument("--allow-agent-host", action="append", default=[])
    parser.add_argument("--allow-environment-host", action="append", default=[])
    parser.add_argument("--agent-env", action="append", default=[])
    parser.add_argument("--agent-kwarg", action="append", default=[])
    parser.add_argument("--agent-include-logs", action="append", default=[])
    parser.add_argument("--agent-exclude-logs", action="append", default=[])
    parser.add_argument(
        "--setup-proxy-from-env",
        action="store_true",
        help=(
            "Forward host proxy env only to setup/install commands via "
            "private HORIZON_HARBOR_SETUP_* agent env; it is filtered before the "
            "agent runner starts."
        ),
    )

    # HorizonHarness runner controls passed as Harbor agent kwargs.
    parser.add_argument(
        "--capabilities",
        default=",".join(DEFAULT_CAPABILITIES),
        help=f"Comma-separated capabilities chosen from {CAPABILITIES}.",
    )
    parser.add_argument("--workflow", choices=WORKFLOWS, default="single")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--provider", choices=["openai", "fake"], default="openai")
    parser.add_argument("--api-kind", default=DEFAULT_API_KIND)
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        default=[],
        help="Harbor model name for the agent. Repeatable.",
    )
    return parser


def _append_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def _append_repeated(cmd: list[str], flag: str, values: list[str]) -> None:
    for value in values:
        cmd.extend([flag, str(value)])


def _validate_args(args: argparse.Namespace) -> None:
    if not any((args.path, args.dataset, args.repo, args.task)):
        raise SystemExit(
            "harbor bench requires one dataset source: --path, --dataset, "
            "--repo, or --task."
        )


def _load_dotenv_for_run(args: argparse.Namespace) -> None:
    if args.provider == "openai" and args.dotenv:
        load_dotenv(args.dotenv)
    if args.harbor_agent == TERMINUS_BASELINE_AGENT:
        token = os.environ.get(OPENAI_AUTH_ENV, "").strip()
        if token and not os.environ.get("OPENAI_API_KEY"):
            # Terminus-2 uses LiteLLM's OpenAI-compatible environment name.
            # Keep the canonical project secret in OPENAI_AUTH_TOKEN and bridge
            # it only in this process; neither command nor manifest contains it.
            os.environ["OPENAI_API_KEY"] = token


def _agent_env_args(args: argparse.Namespace) -> list[str]:
    values: list[str] = list(args.agent_env or [])
    present = {item.split("=", 1)[0] for item in values if "=" in item}
    for key in _PASS_ENV_KEYS:
        value = os.environ.get(key)
        if value and key not in present:
            values.append(f"{key}=${{{key}}}")
            present.add(key)
    for key, value in _DEFAULT_SETUP_ENV.items():
        private_key = f"{_SETUP_ENV_PREFIX}{key}"
        if private_key not in present:
            values.append(f"{private_key}={value}")
            present.add(private_key)
    if args.setup_proxy_from_env:
        for key in _SETUP_PROXY_ENV_KEYS:
            private_key = f"{_SETUP_ENV_PREFIX}{key}"
            value = os.environ.get(key)
            if value and private_key not in present:
                values.append(f"{private_key}=${{{key}}}")
                present.add(private_key)
    return values


def _redact_env_assignment(value: str) -> str:
    if "=" not in value:
        return value
    key, raw = value.split("=", 1)
    if _SENSITIVE_ENV_RE.search(key) and not raw.startswith("${"):
        return f"{key}=****"
    return value


def display_command(command: list[str]) -> list[str]:
    shown: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            shown.append(_redact_env_assignment(part))
            redact_next = False
            continue
        shown.append(part)
        redact_next = part == "--agent-env"
    return shown


def _terminus_models(args: argparse.Namespace) -> list[str]:
    values = list(args.model or [])
    if not values:
        configured = os.environ.get(OPENAI_MODEL_ENV, "").strip()
        if configured:
            values.append(configured)
    if not values:
        raise SystemExit(
            "terminus-2 baseline requires --model or OPENAI_MODEL in the dotenv"
        )
    return [
        value if value.startswith("openai/") else f"openai/{value}" for value in values
    ]


def build_command(args: argparse.Namespace) -> list[str]:
    _validate_args(args)

    cmd = [args.harbor_bin, "run"]
    _append_optional(cmd, "--path", args.path)
    _append_optional(cmd, "--dataset", args.dataset)
    _append_optional(cmd, "--repo", args.repo)
    _append_optional(cmd, "--task", args.task)
    _append_optional(cmd, "--registry-url", args.registry_url)
    _append_optional(cmd, "--registry-path", args.registry_path)
    _append_repeated(cmd, "--include-task-name", args.include_task_names)
    _append_repeated(cmd, "--exclude-task-name", args.exclude_task_names)
    _append_optional(cmd, "--n-tasks", args.n_tasks)

    _append_optional(cmd, "--job-name", args.job_name)
    _append_optional(cmd, "--jobs-dir", args.jobs_dir)
    _append_optional(cmd, "--n-concurrent", args.n_concurrent)
    _append_optional(cmd, "--n-attempts", args.n_attempts)
    _append_optional(cmd, "--max-retries", args.max_retries)
    _append_optional(cmd, "--timeout-multiplier", args.timeout_multiplier)
    _append_optional(cmd, "--agent-timeout-multiplier", args.agent_timeout_multiplier)
    _append_optional(
        cmd, "--agent-setup-timeout-multiplier", args.agent_setup_timeout_multiplier
    )
    if args.quiet:
        cmd.append("--quiet")
    if args.debug:
        cmd.append("--debug")
    if args.install_only:
        cmd.append("--install-only")
    if args.yes:
        cmd.append("--yes")

    _append_optional(cmd, "--env", args.env)
    _append_repeated(cmd, "--allow-agent-host", args.allow_agent_host)
    _append_repeated(cmd, "--allow-environment-host", args.allow_environment_host)

    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        raise SystemExit("--max-output-tokens must be at least 1")

    if args.harbor_agent == TERMINUS_BASELINE_AGENT:
        cmd.extend(["--agent", TERMINUS_BASELINE_AGENT])
        for model in _terminus_models(args):
            cmd.extend(["--model", model])
        agent_kwargs = [
            f"max_turns={args.max_turns}",
            "temperature=1.0",
            "enable_summarize=false",
        ]
        api_base = os.environ.get(OPENAI_BASE_URL_ENV, "").strip()
        if api_base:
            agent_kwargs.append(f"api_base={api_base}")
        if args.max_output_tokens is not None:
            call_kwargs = json.dumps(
                {"max_tokens": args.max_output_tokens}, separators=(",", ":")
            )
            agent_kwargs.append(f"llm_call_kwargs={call_kwargs}")
        reasoning = os.environ.get(REASONING_EFFORT_ENV, "").strip()
        if reasoning:
            agent_kwargs.append(f"reasoning_effort={reasoning}")
    else:
        cmd.extend(["--agent", AGENT_IMPORT_PATH])
        for model in args.model:
            cmd.extend(["--model", model])
        agent_kwargs = [
            f"max_turns={args.max_turns}",
            f"capabilities={args.capabilities}",
            f"workflow={args.workflow}",
            f"provider={args.provider}",
        ]
        if args.max_output_tokens is not None:
            agent_kwargs.append(f"max_output_tokens={args.max_output_tokens}")
        if args.api_kind:
            agent_kwargs.append(f"api_kind={args.api_kind}")
    agent_kwargs.extend(args.agent_kwarg or [])
    _append_repeated(cmd, "--agent-kwarg", agent_kwargs)

    if args.harbor_agent == TERMINUS_BASELINE_AGENT:
        _append_repeated(cmd, "--agent-env", args.agent_env or [])
        include_logs = args.agent_include_logs or []
    else:
        _append_repeated(cmd, "--agent-env", _agent_env_args(args))
        include_logs = [*_AGENT_LOG_INCLUDES, *(args.agent_include_logs or [])]
    _append_repeated(cmd, "--agent-include-logs", include_logs)
    _append_repeated(cmd, "--agent-exclude-logs", args.agent_exclude_logs or [])
    return cmd


def _result_summary(args: argparse.Namespace) -> dict[str, Any] | None:
    jobs_dir = Path(args.jobs_dir)
    job_dir = (
        jobs_dir / args.job_name if args.job_name else find_latest_job_dir(jobs_dir)
    )
    if job_dir is None:
        return None
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        return None
    return summarize_result_file(result_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _load_dotenv_for_run(args)
    cmd = build_command(args)
    jobs_dir = Path(args.jobs_dir)

    if args.dry_run:
        return {
            "bench": NAME,
            "status_code": 0,
            "command": display_command(cmd),
            "jobs_dir": str(jobs_dir),
            "job_name": args.job_name,
            "result_path": str(jobs_dir / args.job_name / "result.json"),
            "run_dir": str(jobs_dir / args.job_name),
        }

    jobs_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        status_code = proc.returncode
    except FileNotFoundError as exc:
        return {
            "bench": NAME,
            "status_code": 127,
            "command": cmd,
            "jobs_dir": str(jobs_dir),
            "job_name": args.job_name,
            "error": str(exc),
        }

    summary = _result_summary(args)
    result: dict[str, Any] = {
        "bench": NAME,
        "status_code": status_code,
        "command": cmd,
        "jobs_dir": str(jobs_dir),
        "job_name": args.job_name,
        "result_path": None,
        "run_dir": str(jobs_dir / args.job_name),
        "summary": None,
    }
    if summary is not None:
        result["result_path"] = summary["result_path"]
        result["summary"] = summary
    return result


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    raise SystemExit(run(parser.parse_args(argv)).get("status_code", 0))


if __name__ == "__main__":
    main()
