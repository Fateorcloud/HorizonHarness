"""Capture one benchmark launch without owning benchmark execution.

The unified ``runs.run_bench`` entry point calls this module before it invokes
an existing benchmark adapter.  The manifest is first written to a shared
sidecar directory, then copied into the adapter's existing run directory after
the adapter returns.  No task execution, scoring, or failure classification is
duplicated here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from argparse import Namespace
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from horizon_harness.evals.benchmark import build_run_manifest, write_manifest
from horizon_harness.llm.env import load_dotenv


_TASK_FIELDS = (
    "dataset",
    "dataset_name",
    "variant",
    "instance_id",
    "instance_json",
    "path",
    "repo",
    "task",
    "include_task_names",
    "exclude_task_names",
    "n_tasks",
    "ids_file",
    "all",
    "filter",
    "slice",
    "limit",
)
_BUDGET_FIELDS = (
    "max_turns",
    "max_output_tokens",
    "wall_time_seconds",
    "timeout",
    "concurrency",
    "parallel",
    "n_concurrent",
    "n_attempts",
    "max_retries",
    "timeout_multiplier",
    "agent_timeout_multiplier",
    "agent_setup_timeout_multiplier",
)
_RUNTIME_FIELDS = (
    "platform",
    "network_mode",
    "image_tag",
    "pull",
    "cpus",
    "mem_limit",
    "in_env_scoring",
    "no_scoring",
    "security_opt",
)
_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _selected(args: Namespace, names: tuple[str, ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in names:
        value = getattr(args, name, None)
        if value not in (None, "", [], ()):
            values[name] = _json_value(value)
    return values


def _git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = revision.stdout.strip() if revision.returncode == 0 else "unavailable"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return commit, bool(status.stdout.strip()) if status.returncode == 0 else True


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _run_id(bench: str, args: Namespace, started_at: datetime) -> str:
    for name in ("job_name", "run_id"):
        value = str(getattr(args, name, "") or "").strip()
        if value:
            return value
    return f"{bench}-{started_at.strftime('%Y%m%d-%H%M%S')}"


def _intended_artifact_root(
    root: Path, bench: str, run_id: str, args: Namespace
) -> str:
    jobs_dir = getattr(args, "jobs_dir", None)
    run_root = getattr(args, "run_root", None)
    if jobs_dir:
        target = Path(jobs_dir) / run_id
    elif run_root:
        target = Path(run_root) / run_id
    else:
        target = root / "evals/out" / bench / run_id
    if not target.is_absolute():
        target = root / target
    try:
        return str(target.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(target.resolve())


def _provider_manifest(args: Namespace, env: Mapping[str, str]) -> dict[str, Any]:
    provider = str(getattr(args, "provider", "openai") or "openai")
    requested_id = (env.get("OPENAI_MODEL") or "").strip() or provider
    base_url = (env.get("OPENAI_BASE_URL") or "").strip()
    endpoint_host = urlsplit(base_url).hostname if base_url else None
    api = str(getattr(args, "api_kind", "") or "").strip()
    if not api:
        api = (env.get("HORIZON_API_KIND") or "openai-chat").strip()
    return {
        "requested_id": requested_id,
        "provider": provider,
        "api": api,
        "endpoint": base_url or None,
        "endpoint_host": endpoint_host,
        "reasoning": (env.get("HORIZON_REASONING_EFFORT") or "").strip() or None,
    }


def _agent_manifest(args: Namespace, env: Mapping[str, str]) -> dict[str, Any]:
    harbor_agent = str(getattr(args, "harbor_agent", "") or "").strip()
    if harbor_agent == "terminus-2":
        return {
            "name": "terminus-2",
            "capabilities": ["terminal"],
            "workflow": "terminus-2",
            "context_summarization": False,
            "memory": False,
            "skills": False,
            "mcp": False,
        }
    raw_capabilities = (
        str(getattr(args, "capabilities", "") or "").strip()
        or (env.get("HORIZON_CAPABILITIES") or "").strip()
        or "bash,read,task"
    )
    workflow = (
        str(getattr(args, "workflow", "") or "").strip()
        or (env.get("HORIZON_WORKFLOW") or "").strip()
        or "single"
    )
    payload = {
        "capabilities": [
            part.strip() for part in raw_capabilities.split(",") if part.strip()
        ],
        "workflow": workflow,
        "memory": False,
        "skills": False,
        "mcp": False,
    }
    if harbor_agent:
        payload["name"] = "horizon-harness"
    return payload


def _runtime_manifest(args: Namespace) -> dict[str, Any]:
    versions = {
        name: value
        for name, value in (
            ("horizon-harness", _version("horizon-harness")),
            ("harbor", _version("harbor")),
            ("swebench", _version("swebench")),
            ("programbench", _version("programbench")),
        )
        if value is not None
    }
    return {
        **_selected(args, _RUNTIME_FIELDS),
        "python": sys.version.split()[0],
        "packages": versions,
    }


@dataclass(frozen=True)
class RunManifestHandle:
    """A launch manifest already persisted before its adapter starts."""

    payload: dict[str, Any]
    sidecar_path: Path

    def attach(self, outcome: Mapping[str, Any]) -> dict[str, Any]:
        """Copy the frozen manifest into an existing adapter artifact directory."""

        result = dict(outcome)
        run_dir_value = result.get("run_dir")
        run_dir = Path(str(run_dir_value)) if run_dir_value else None
        if run_dir is not None and run_dir.is_dir():
            attached = run_dir / "run-manifest.json"
            write_manifest(attached, self.payload)
            manifest_path = attached
        else:
            manifest_path = self.sidecar_path
        result["run_manifest_path"] = str(manifest_path)
        return result


def start_run_manifest(
    bench: str,
    args: Namespace,
    *,
    root: Path,
    environ: MutableMapping[str, str] | None = None,
    started_at: datetime | None = None,
    git_state: tuple[str, bool] | None = None,
) -> RunManifestHandle:
    """Freeze safe launch inputs and persist them before a benchmark runs."""

    env = os.environ if environ is None else environ
    dotenv = str(getattr(args, "dotenv", "") or "").strip()
    if str(getattr(args, "provider", "openai")) == "openai" and dotenv:
        load_dotenv(dotenv, environ=env)
    now = started_at or datetime.now(timezone.utc)
    run_id = _run_id(bench, args, now)
    commit, dirty = git_state or _git_state(root)
    artifact_root = _intended_artifact_root(root, bench, run_id, args)
    task_selection = _selected(args, _TASK_FIELDS)
    payload = build_run_manifest(
        run_id=run_id,
        commit=commit,
        dirty=dirty,
        benchmark={"name": bench},
        task_selection=task_selection,
        model=_provider_manifest(args, env),
        agent=_agent_manifest(args, env),
        budget=_selected(args, _BUDGET_FIELDS),
        runtime=_runtime_manifest(args),
        artifact_root=artifact_root,
        run_at=now.isoformat(),
    )
    slug = _SAFE_SLUG_RE.sub("-", run_id).strip("-.") or "run"
    sidecar = root / "evals/out/run-manifests" / f"{bench}-{slug}.json"
    write_manifest(sidecar, payload)
    return RunManifestHandle(payload=payload, sidecar_path=sidecar)


__all__ = ["RunManifestHandle", "start_run_manifest"]
