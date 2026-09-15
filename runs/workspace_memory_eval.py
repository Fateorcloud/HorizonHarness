"""Freeze and execute the Task Workspace / indexed-memory SWE-bench matrix.

``plan`` is offline and never calls a model. ``run`` is the explicit paid entry:
it uses the frozen four tasks, a conservative token ledger, fresh instance-image
checkouts, safe-turn checkpoints, the fixed S/W/K/Y order, and the official
SWE-bench verifier. Oracle rows are read only after an Agent run and are never
mounted into its command container or copied into its checkout.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from filelock import FileLock

from horizon_harness.llm import LLMRequest, LLMResponse, complete
from horizon_harness.llm.env import (
    load_dotenv,
    provider_from_env,
    request_extra_from_env,
)
from horizon_harness.memory import (
    IndexedMemory,
    ModelMemoryExtractor,
    SentenceTransformerEmbedder,
)
from horizon_harness.spec import AgentSpec
from horizon_harness.workspace import (
    ModelTaskSummarizer,
    ModelWorkspaceProjector,
    TaskSession,
    WorkspaceStore,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    ROOT / "evals/out/task-workspace-memory-delivery/swebench/task-selection.json"
)
DEFAULT_OUTPUT = ROOT / "evals/out/task-workspace-memory-formal"
TOTAL_BUDGET = 5_000_000
PHASE_LIMITS = {"development": 1_000_000, "setup": 1_000_000, "suffix": 3_000_000}
PREFIX_TURNS = 8
SUFFIX_TURNS = 32
TASK_WALL_SECONDS = 30 * 60
CELL_ORDER = (
    ("S-H1", "H1", "summary"),
    ("W-H1", "H1", "workspace"),
    ("W-H2", "H2", "workspace"),
    ("S-H2", "H2", "summary"),
    ("S-H3", "H3", "summary"),
    ("W-H3", "H3", "workspace"),
    ("K-H1", "H1", "keyword"),
    ("Y-H1", "H1", "hybrid"),
    ("Y-H2", "H2", "hybrid"),
    ("K-H2", "H2", "keyword"),
)
MEMORY_SCOPE = "github.com/pytest-dev/pytest"


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenTask:
    role: str
    instance_id: str
    base_commit: str
    image: str
    task_file: Path
    verifier_file: Path


class TokenLedger:
    """Process-safe conservative reservation and usage ledger."""

    def __init__(
        self,
        root: Path,
        *,
        total_limit: int = TOTAL_BUDGET,
        complete_fn: Callable[[LLMRequest], LLMResponse] = complete,
        initial_charged: int = 0,
    ) -> None:
        self.root = root.resolve()
        self.total_limit = total_limit
        self.state_path = self.root / "budget.json"
        self.usage_path = self.root / "usage.jsonl"
        self.lock = FileLock(str(self.root / ".budget.lock"))
        self.thread_lock = threading.Lock()
        self.complete_fn = complete_fn
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            phases = {key: 0 for key in PHASE_LIMITS}
            phases["development"] = initial_charged
            self._write_state({"charged": initial_charged, "phases": phases})

    def complete(self, phase: str, label: str) -> Callable[[LLMRequest], LLMResponse]:
        if phase not in PHASE_LIMITS:
            raise ValueError(f"unknown budget phase {phase!r}")

        def invoke(request: LLMRequest) -> LLMResponse:
            if phase == "setup" and not label.startswith("prefix-"):
                extra = dict(request.extra)
                if request.provider.model.startswith("glm-"):
                    extra["extra_body"] = {"thinking": {"type": "enabled"}}
                    extra["reasoning_effort"] = "low"
                if label.startswith("workspace-") or "extraction" in label:
                    extra["response_format"] = {"type": "json_object"}
                request = dataclasses.replace(request, max_tokens=8192, extra=extra)
            reservation = _request_reservation(request)
            call_id = uuid.uuid4().hex
            self._reserve(phase, reservation)
            started = time.monotonic()
            try:
                response = self.complete_fn(request)
            except Exception as exc:
                self._settle(
                    phase,
                    reservation,
                    reservation,
                    {
                        "call_id": call_id,
                        "phase": phase,
                        "label": label,
                        "status": "error_unknown_usage",
                        "reservation": reservation,
                        "charged": reservation,
                        "elapsed_seconds": time.monotonic() - started,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise
            known = response.usage.context_tokens > 0
            _write_json(
                self.root / "responses" / f"{call_id}.json",
                {
                    "label": label,
                    "text": response.text,
                    "stop_reason": response.stop_reason,
                    "usage": dataclasses.asdict(response.usage),
                    "model": response.model,
                },
            )
            charged = response.usage.context_tokens if known else reservation
            self._settle(
                phase,
                reservation,
                charged,
                {
                    "call_id": call_id,
                    "phase": phase,
                    "label": label,
                    "status": "ok" if known else "ok_unknown_usage",
                    "reservation": reservation,
                    "charged": charged,
                    "usage": dataclasses.asdict(response.usage),
                    "model": response.model,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            return response

        return invoke

    def _reserve(self, phase: str, amount: int) -> None:
        with self.thread_lock, self.lock:
            state = self._read_state()
            phase_used = int(state["phases"][phase])
            charged = int(state["charged"])
            if phase_used + amount > PHASE_LIMITS[phase]:
                raise BudgetExhausted(
                    f"{phase} reservation exceeds {PHASE_LIMITS[phase]} tokens"
                )
            if charged + amount > self.total_limit:
                raise BudgetExhausted(
                    f"total reservation exceeds {self.total_limit} tokens"
                )
            state["charged"] = charged + amount
            state["phases"][phase] = phase_used + amount
            self._write_state(state)

    def _settle(
        self,
        phase: str,
        reservation: int,
        charged: int,
        row: Mapping[str, Any],
    ) -> None:
        with self.thread_lock, self.lock:
            state = self._read_state()
            difference = charged - reservation
            state["charged"] = int(state["charged"]) + difference
            state["phases"][phase] = int(state["phases"][phase]) + difference
            self._write_state(state)
            with self.usage_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                stream.write("\n")

    def _read_state(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]", json.loads(self.state_path.read_text(encoding="utf-8"))
        )

    def _write_state(self, state: Mapping[str, Any]) -> None:
        _write_json(self.state_path, state)


def _request_reservation(request: LLMRequest) -> int:
    wire = {
        "system_prompt": request.system_prompt,
        "messages": [dataclasses.asdict(message) for message in request.messages],
        "tools": [dataclasses.asdict(tool) for tool in request.tools],
        "extra": request.extra,
    }
    input_estimate = (
        len(json.dumps(wire, ensure_ascii=False, default=repr)) + 2
    ) // 3 + 64
    output = request.max_tokens or request.provider.default_max_tokens or 2_048
    return input_estimate + int(output)


def build_plan(
    selection_path: Path, output_dir: Path, *, implementation: Path = ROOT
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    base = selection_path.parent
    rows = selection.get("instances")
    if not isinstance(rows, list):
        raise ValueError("selection instances must be a list")
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = cast("dict[str, Any]", raw)
        role = str(row["role"])
        if role in seen or role not in {"D", "H1", "H2", "H3"}:
            raise ValueError(f"invalid or duplicate task role {role!r}")
        seen.add(role)
        task_file = (base / str(row["agent_task_file"])).resolve()
        verifier_file = (base / str(row["verifier_row_file"])).resolve()
        task_payload = json.loads(task_file.read_text(encoding="utf-8"))
        forbidden = {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"}
        leaked = forbidden.intersection(task_payload)
        if leaked:
            raise ValueError(f"agent task file leaks verifier fields: {sorted(leaked)}")
        if task_file.parent.name != "tasks" or verifier_file.parent.name != "verifier":
            raise ValueError(
                "task and verifier inputs must remain in separate directories"
            )
        tasks.append(
            {
                "role": role,
                "instance_id": str(row["instance_id"]),
                "base_commit": str(row["base_commit"]),
                "image": str(row["instance_image_key"]),
                "task_file": str(task_file),
                "task_sha256": _sha256(task_file),
                "verifier_file": str(verifier_file),
                "verifier_sha256": _sha256(verifier_file),
            }
        )
    if seen != {"D", "H1", "H2", "H3"}:
        raise ValueError("selection must contain exactly D, H1, H2, and H3")
    delivery_manifest = selection_path.parent.parent / "manifest.json"
    starting_known_tokens = 0
    if delivery_manifest.is_file():
        delivery = json.loads(delivery_manifest.read_text(encoding="utf-8"))
        starting_known_tokens = int(
            delivery.get("budget", {}).get("known_remote_tokens", 0)
        )
    plan = {
        "schema": "horizon-harness.workspace-memory-formal.v1",
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "dataset": selection["dataset"],
        "split": selection["split"],
        "dataset_revision": selection["dataset_revision"],
        "implementation": str(implementation.resolve()),
        "implementation_fingerprint": _implementation_fingerprint(implementation),
        "tasks": tasks,
        "development_attempt_limit": 3,
        "prefix_turn_limit": PREFIX_TURNS,
        "suffix_turn_limit": SUFFIX_TURNS,
        "logical_path_turn_limit": PREFIX_TURNS + SUFFIX_TURNS,
        "wall_time_seconds": TASK_WALL_SECONDS,
        "suffix_order": [
            {"cell": cell, "role": role, "representation": mode}
            for cell, role, mode in CELL_ORDER
        ],
        "parallel_suffix_limit": 2,
        "memory_scope": MEMORY_SCOPE,
        "budget": {"total": TOTAL_BUDGET, "phases": PHASE_LIMITS},
        "starting_known_tokens": starting_known_tokens,
        "retry_policy": "disabled_for_formal_calls",
        "oracle_isolation": (
            "verifier_file is host-only and is read after Agent execution; it is "
            "never mounted or copied into an Agent checkout"
        ),
        "status": "frozen_not_started",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "experiment.json", plan)
    return plan


def load_tasks(plan: Mapping[str, Any]) -> dict[str, FrozenTask]:
    tasks: dict[str, FrozenTask] = {}
    for raw in cast("list[dict[str, Any]]", plan["tasks"]):
        task = FrozenTask(
            role=str(raw["role"]),
            instance_id=str(raw["instance_id"]),
            base_commit=str(raw["base_commit"]),
            image=str(raw["image"]),
            task_file=Path(str(raw["task_file"])),
            verifier_file=Path(str(raw["verifier_file"])),
        )
        tasks[task.role] = task
    return tasks


class FormalRunner:
    def __init__(
        self, plan: Mapping[str, Any], output_dir: Path, *, dotenv: Path
    ) -> None:
        self.plan = plan
        self.output_dir = output_dir.resolve()
        if self.output_dir in (Path("/"), Path.home().resolve()):
            raise ValueError(
                "formal output directory must not be a filesystem root or home"
            )
        self.tasks = load_tasks(plan)
        load_dotenv(dotenv)
        self.provider = provider_from_env(read_reasoning=True)
        self.request_extra = request_extra_from_env(env=os.environ)
        self.ledger = TokenLedger(
            self.output_dir,
            initial_charged=int(plan.get("starting_known_tokens", 0)),
        )
        self.spec = AgentSpec(
            name="workspace-agent",
            role="Complete the repository task with bash, read, and edit tools.",
            capabilities=("bash", "read", "edit"),
            workflow="single",
        )

    def run(
        self,
        *,
        development_attempts: int,
        embedding_revision: str,
        recovery: bool = False,
        workspace_only: bool = False,
    ) -> None:
        if not 1 <= development_attempts <= 3:
            raise ValueError("development_attempts must be between 1 and 3")
        if self.provider.api == "fake":
            raise ValueError("formal evaluation requires a real provider")
        if workspace_only:
            self._assert_frozen_inputs()
            self.workspace_only = True
            results = []
            self._write_results(results)
            for role in ("D", "H1", "H2", "H3"):
                cell = f"W-{role}-workspace-only"
                try:
                    prefix = self._prefix(self.tasks[role])
                    row = self._suffix(
                        cell,
                        self.tasks[role],
                        prefix,
                        mode="workspace",
                        memory_db=self.output_dir / "memory/unused.sqlite3",
                        embedding_revision=embedding_revision,
                    )
                except Exception as exc:
                    row = {
                        "cell": cell,
                        "role": role,
                        "verdict": "interrupted",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _write_json(self.output_dir / "cells" / cell / "failure.json", row)
                results.append(row)
                self._write_results(results)
            _write_json(
                self.output_dir / "workspace-only.json",
                {"results": results, "note": "Workspace-only four-task validation"},
            )
            return
        if recovery:
            _write_json(
                self.output_dir / "recovery.json",
                {
                    "original_fingerprint": self.plan["implementation_fingerprint"],
                    "recovery_fingerprint": _implementation_fingerprint(ROOT),
                    "auxiliary_output_cap": 8192,
                    "note": "Explicit recovery; prior costs retained; not original frozen scores.",
                },
            )
            for row in self.plan["tasks"]:
                for kind in ("task", "verifier"):
                    if _sha256(Path(row[f"{kind}_file"])) != row[f"{kind}_sha256"]:
                        raise ValueError("frozen task input changed")
        else:
            self._assert_frozen_inputs()
        self._write_results([])
        _write_json(
            self.output_dir / "runtime.json",
            {
                "provider": {
                    "id": self.provider.id,
                    "api": self.provider.api,
                    "model": self.provider.model,
                    "endpoint_host": (
                        urlparse(self.provider.base_url).hostname
                        if self.provider.base_url
                        else None
                    ),
                    "reasoning": self.provider.default_reasoning,
                    "context_window": self.provider.context_window,
                },
                "credentials_recorded": False,
                "development_attempts": development_attempts,
                "embedding_revision": embedding_revision,
            },
        )
        memory_db = self.output_dir / "memory/frozen.sqlite3"
        d_result = None
        for attempt in range(1, development_attempts + 1):
            d_result = self._fresh_run(
                self.tasks["D"],
                label=f"D-{attempt}",
                phase="development",
                max_turns=20,
            )
        assert d_result is not None
        d_session, d_run, d_patch, d_verdict = d_result
        memory = IndexedMemory(memory_db)
        extraction_session = TaskSession(
            self.provider,
            code_dir=d_run["workdir"],
            store_dir=d_run["store"],
            spec=self.spec,
            request_extra=self.request_extra,
            memory=memory,
            memory_scope=str(self.plan["memory_scope"]),
            memory_extractor=ModelMemoryExtractor(
                self.provider,
                request_extra=self.request_extra,
                complete_fn=self.ledger.complete("setup", "D-memory-extraction"),
            ),
            complete_fn=self.ledger.complete("development", "D-finalization"),
        )
        extraction_session.record_verifier(
            d_session.task_id,
            command="official SWE-bench verifier",
            exit_code=0 if d_verdict == "pass" else 1,
            evidence_ref=str(d_run["verifier_result"]),
        )
        try:
            ids, extraction = extraction_session.finalize_memory(
                d_session.task_id,
                outcome="completed" if d_verdict == "pass" else "failed",
            )
            _write_json(
                self.output_dir / "memory/extraction.json",
                {
                    "entry_ids": ids,
                    "usage": dataclasses.asdict(extraction.usage),
                    "model": extraction.model,
                },
            )
        except Exception as exc:
            _write_json(
                self.output_dir / "memory/extraction.json",
                {"entry_ids": [], "error": f"{type(exc).__name__}: {exc}"},
            )

        self._write_results([])
        prefixes = {role: self._prefix(self.tasks[role]) for role in ("H1", "H2", "H3")}
        results: list[dict[str, Any]] = []
        for cell, role, mode in CELL_ORDER:
            budget_exhausted = False
            try:
                row = self._suffix(
                    cell,
                    self.tasks[role],
                    prefixes[role],
                    mode=mode,
                    memory_db=memory_db,
                    embedding_revision=embedding_revision,
                )
            except Exception as exc:
                budget_exhausted = isinstance(exc, BudgetExhausted)
                row = {
                    "cell": cell,
                    "role": role,
                    "instance_id": self.tasks[role].instance_id,
                    "representation": mode,
                    "verdict": (
                        "budget_exhausted" if budget_exhausted else "infra_error"
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _write_json(self.output_dir / "cells" / cell / "result.json", row)
            results.append(row)
            self._write_results(results)
            if budget_exhausted:
                break

    def _write_results(self, results: Sequence[Mapping[str, Any]]) -> None:
        _write_json(
            self.output_dir / "results.json",
            {
                "schema": "horizon-harness.workspace-memory-results.v1",
                "cells": list(results),
            },
        )
        (self.output_dir / "report.md").write_text(
            "# Recovery evaluation\n\nImplementation repaired after a failed run; "
            "these results are not the original frozen comparison.\n\n"
            + json.dumps(list(results), indent=2)
            + "\n",
            encoding="utf-8",
        )

    def _prefix(self, task: FrozenTask) -> dict[str, Any]:
        label = f"prefix-{task.role}"
        run_root = self.output_dir / "runs" / label
        workdir = run_root / "workdir"
        store_path = run_root / "store"
        saved = list(store_path.glob("*/task.json"))
        record = None
        remaining = PREFIX_TURNS
        if saved:
            store = WorkspaceStore(store_path)
            record = store.assert_resumable(saved[0].parent.name)
            # Count persisted complete turns across all resumed segments.
            traces = store.task_dir(record.task_id) / "traces"
            completed = sum(
                1
                for path in traces.glob("*.jsonl")
                for line in path.read_text().splitlines()
                if json.loads(line).get("kind") == "turn_end"
            )
            remaining = max(0, PREFIX_TURNS - completed)
            if not remaining:
                return self._project_prefix(
                    task, record, store.load_messages(record), workdir, store_path
                )
        else:
            _checkout_from_image(task.image, workdir)
        payload = json.loads(task.task_file.read_text(encoding="utf-8"))
        task_text = str(payload["problem_statement"])
        container = _start_container(task.image, workdir, label)
        try:
            runner = TaskSession(
                self.provider,
                code_dir=workdir,
                store_dir=store_path,
                spec=self.spec,
                request_extra=self.request_extra,
                bash_exec_prefix=_container_prefix(container),
                complete_fn=self.ledger.complete("setup", label),
                context_budget=24_000,
            )
            if record is not None:
                session = runner.resume(
                    record.task_id, max_turns=remaining, pause_after_turns=remaining
                )
            else:
                session = runner.start(
                    task_text,
                    constraints=("Do not access verifier or oracle data.",),
                    acceptance=(
                        "Produce a repository patch for independent verification.",
                    ),
                    max_turns=PREFIX_TURNS,
                    pause_after_turns=PREFIX_TURNS,
                )
        finally:
            _stop_container(container)
        return self._project_prefix(
            task, session.record, session.state.messages, workdir, store_path
        )

    def _project_prefix(self, task, record, messages, workdir, store_path):
        store = WorkspaceStore(store_path)
        snapshot = self.output_dir / "snapshots" / task.role
        if getattr(self, "workspace_only", False):
            marker = snapshot / "prefix.json"
            if not marker.exists():
                _copy_tree(workdir, snapshot / "workdir")
                _copy_tree(store_path, snapshot / "store")
                _write_json(
                    marker, {"task_id": record.task_id, "revision": record.revision}
                )
            return {
                "task_id": record.task_id,
                "snapshot": snapshot,
                "workdir": workdir,
                "store": store_path,
                "resumable": record.checkpoint.resumable,
            }
        if (snapshot / "summary.txt").exists() and (snapshot / "store").exists():
            return {
                "task_id": record.task_id,
                "snapshot": snapshot,
                "workdir": workdir,
                "store": store_path,
                "resumable": record.checkpoint.resumable,
            }
        _copy_tree(workdir, snapshot / "workdir")
        projected = ModelWorkspaceProjector(
            self.provider,
            request_extra=self.request_extra,
            complete_fn=self.ledger.complete("setup", f"workspace-{task.role}"),
        ).project(record, messages, code_dir=workdir)
        store.save(projected.record)
        _copy_tree(store_path, snapshot / "store")
        _write_json(
            snapshot / "workspace-usage.json",
            dataclasses.asdict(projected.usage),
        )
        summary = ModelTaskSummarizer(
            self.provider,
            request_extra=self.request_extra,
            complete_fn=self.ledger.complete("setup", f"summary-{task.role}"),
        ).summarize(record.task, messages)
        (snapshot / "summary.txt").write_text(summary.text, encoding="utf-8")
        _write_json(snapshot / "summary-usage.json", dataclasses.asdict(summary.usage))
        return {
            "task_id": record.task_id,
            "snapshot": snapshot,
            "workdir": workdir,
            "store": store_path,
            "resumable": record.checkpoint.resumable,
        }

    def _suffix(
        self,
        cell: str,
        task: FrozenTask,
        prefix: Mapping[str, Any],
        *,
        mode: str,
        memory_db: Path,
        embedding_revision: str,
    ) -> dict[str, Any]:
        cell_root = self.output_dir / "cells" / cell
        if (cell_root / "result.json").exists():
            return json.loads((cell_root / "result.json").read_text())
        if not bool(prefix["resumable"]):
            row = {
                "cell": cell,
                "role": task.role,
                "instance_id": task.instance_id,
                "representation": mode,
                "verdict": "no_continuation",
            }
            _write_json(cell_root / "result.json", row)
            return row
        workdir = Path(prefix["workdir"])
        cell_root.mkdir(parents=True, exist_ok=True)
        store = Path(prefix["store"])
        snapshot = Path(prefix["snapshot"])
        _copy_tree(snapshot / "workdir", workdir)
        _copy_tree(snapshot / "store", store)
        container = _start_container(task.image, workdir, cell)
        started = time.monotonic()
        try:
            memory = None
            representation = mode if mode in ("summary", "combined") else "workspace"
            summary = (
                (snapshot / "summary.txt").read_text(encoding="utf-8")
                if representation in ("summary", "combined")
                else ""
            )
            if mode == "keyword":
                memory = IndexedMemory(memory_db, mode="keyword")
            elif mode == "hybrid":
                embedder = SentenceTransformerEmbedder(revision=embedding_revision)
                memory = IndexedMemory(memory_db, mode="hybrid", embedder=embedder)
                memory.rebuild_vectors()
            session = TaskSession(
                self.provider,
                code_dir=workdir,
                store_dir=store,
                spec=self.spec,
                request_extra=self.request_extra,
                memory=memory,
                memory_scope=str(self.plan["memory_scope"]),
                bash_exec_prefix=_container_prefix(container),
                complete_fn=self.ledger.complete("suffix", cell),
                context_representation=representation,
                continuation_summary=summary,
                summary_revision=(
                    WorkspaceStore(store).load(str(prefix["task_id"])).revision
                    if mode == "combined"
                    else None
                ),
                context_budget=24_000 if cell.endswith("workspace-only") else 12_000,
            )
            result = session.resume(
                str(prefix["task_id"]),
                max_turns=SUFFIX_TURNS,
                abort=lambda: time.monotonic() - started >= TASK_WALL_SECONDS,
            )
        finally:
            _stop_container(container)
        patch = _git_patch(workdir, task.base_commit)
        patch_path = cell_root / "model.patch"
        patch_path.write_text(patch, encoding="utf-8")
        verdict, verifier_result = _verify(task, patch, cell_root)
        row = {
            "cell": cell,
            "role": task.role,
            "instance_id": task.instance_id,
            "representation": mode,
            "verdict": verdict,
            "task_id": result.task_id,
            "run_id": result.run_id,
            "workspace_status": result.status,
            "patch": str(patch_path),
            "verifier_result": str(verifier_result),
        }
        _copy_tree(store, cell_root / "workspace-evidence")
        _write_json(cell_root / "result.json", row)
        return row

    def _fresh_run(
        self,
        task: FrozenTask,
        *,
        label: str,
        phase: str,
        max_turns: int,
        pause_after_turns: int | None = None,
        verify: bool = True,
    ):
        run_root = self.output_dir / "runs" / label
        workdir = run_root / "workdir"
        store = run_root / "store"
        existing = run_root / "result.json"
        if existing.exists():
            row = json.loads(existing.read_text(encoding="utf-8"))
            return (
                SimpleNamespace(**row),
                {
                    "workdir": workdir,
                    "store": store,
                    "patch": run_root / "model.patch",
                    "verifier_result": run_root / "verifier",
                },
                (run_root / "model.patch").read_text(),
                row["verdict"],
            )
        _checkout_from_image(task.image, workdir)
        container = _start_container(task.image, workdir, label)
        payload = json.loads(task.task_file.read_text(encoding="utf-8"))
        try:
            session = TaskSession(
                self.provider,
                code_dir=workdir,
                store_dir=store,
                spec=self.spec,
                request_extra=self.request_extra,
                bash_exec_prefix=_container_prefix(container),
                complete_fn=self.ledger.complete(phase, label),
            ).start(
                str(payload["problem_statement"]),
                constraints=("Do not access verifier or oracle data.",),
                acceptance=(
                    "Produce a repository patch for independent verification.",
                ),
                max_turns=max_turns,
                pause_after_turns=pause_after_turns,
            )
        finally:
            _stop_container(container)
        patch = _git_patch(workdir, task.base_commit)
        patch_path = run_root / "model.patch"
        patch_path.write_text(patch, encoding="utf-8")
        verdict, verifier_result = (
            _verify(task, patch, run_root)
            if verify
            else ("not_run", run_root / "verifier")
        )
        paths = {
            "workdir": workdir,
            "store": store,
            "patch": patch_path,
            "verifier_result": verifier_result,
        }
        _write_json(
            run_root / "result.json",
            {
                "label": label,
                "instance_id": task.instance_id,
                "task_id": session.task_id,
                "run_id": session.run_id,
                "status": session.status,
                "verdict": verdict,
            },
        )
        return session, paths, patch, verdict

    def _assert_frozen_inputs(self) -> None:
        if (
            _implementation_fingerprint(Path(str(self.plan["implementation"])))
            != self.plan["implementation_fingerprint"]
        ):
            raise ValueError("implementation fingerprint changed after plan freeze")
        for row in cast("list[dict[str, Any]]", self.plan["tasks"]):
            for kind in ("task", "verifier"):
                path = Path(str(row[f"{kind}_file"]))
                if _sha256(path) != row[f"{kind}_sha256"]:
                    raise ValueError(f"frozen {kind} input changed: {path}")


def _checkout_from_image(image: str, target: Path) -> None:
    _fresh_directory(target)
    container = f"workspace-export-{uuid.uuid4().hex[:12]}"
    subprocess.run(
        ["docker", "create", "--name", container, image],
        check=True,
        capture_output=True,
    )
    try:
        subprocess.run(
            ["docker", "cp", f"{container}:/testbed/.", str(target)], check=True
        )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], check=False, capture_output=True
        )


def _start_container(image: str, workdir: Path, label: str) -> str:
    name = (
        "workspace-eval-"
        + hashlib.sha1(f"{label}-{uuid.uuid4()}".encode()).hexdigest()[:12]
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            "-e",
            "GIT_CONFIG_COUNT=1",
            "-e",
            "GIT_CONFIG_KEY_0=safe.directory",
            "-e",
            "GIT_CONFIG_VALUE_0=/testbed",
            "-v",
            f"{workdir.resolve()}:/testbed",
            "-w",
            "/testbed",
            image,
            "tail",
            "-f",
            "/dev/null",
        ],
        check=True,
        capture_output=True,
    )
    return name


def _container_prefix(name: str) -> tuple[str, ...]:
    return ("docker", "exec", "-w", "/testbed", name, "conda", "run", "-n", "testbed")


def _stop_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)


def _verify(task: FrozenTask, patch: str, root: Path) -> tuple[str, Path]:
    verifier_root = root.resolve() / "verifier"
    verifier_root.mkdir(parents=True, exist_ok=True)
    predictions = (verifier_root / "predictions.jsonl").resolve()
    predictions.write_text(
        json.dumps(
            {
                "instance_id": task.instance_id,
                "model_patch": patch,
                "model_name_or_path": "horizon-workspace-memory",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_dir = (verifier_root / "reports").resolve()
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(task.verifier_file),
        "--split",
        "test",
        "--instance_ids",
        task.instance_id,
        "--predictions_path",
        str(predictions),
        "--max_workers",
        "1",
        "--timeout",
        "600",
        "--cache_level",
        "instance",
        "--clean",
        "False",
        "--run_id",
        root.name,
        "--report_dir",
        str(report_dir),
    ]
    try:
        completed = subprocess.run(
            command, cwd=verifier_root, text=True, capture_output=True, timeout=660
        )
    except subprocess.TimeoutExpired as exc:
        (verifier_root / "stdout.log").write_text(
            _subprocess_output(exc.stdout), encoding="utf-8"
        )
        (verifier_root / "stderr.log").write_text(
            _subprocess_output(exc.stderr), encoding="utf-8"
        )
        result = verifier_root / "result.json"
        _write_json(
            result,
            {"verdict": "timeout", "exit_code": None, "reports": []},
        )
        return "timeout", result
    (verifier_root / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (verifier_root / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    reports = [
        *report_dir.glob("*.json"),
        *verifier_root.glob("horizon-workspace-memory.*.json"),
    ]
    verdict = _verifier_verdict(task.instance_id, reports, completed.returncode)
    result = verifier_root / "result.json"
    _write_json(
        result,
        {
            "verdict": verdict,
            "exit_code": completed.returncode,
            "reports": [str(p) for p in reports],
        },
    )
    return verdict, result


def _verifier_verdict(instance_id: str, reports: Sequence[Path], exit_code: int) -> str:
    """Classify an official report without treating missing evidence as a failure."""

    completed = False
    for report in reports:
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if instance_id in set(data.get("resolved_ids") or ()):
            return "pass"
        if instance_id in set(data.get("empty_patch_ids") or ()):
            return "fail" if exit_code == 0 else "infra_error"
        completed = completed or instance_id in set(data.get("completed_ids") or ())
    if exit_code != 0 or not completed:
        return "infra_error"
    return "fail"


def _subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _git_patch(workdir: Path, base_commit: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workdir), "-c", "core.fileMode=false", "diff", base_commit],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _provider_fingerprint(provider) -> str:
    safe = {
        "id": provider.id,
        "api": provider.api,
        "model": provider.model,
        "base_url": provider.base_url,
        "reasoning": provider.default_reasoning,
        "default_max_tokens": min(provider.default_max_tokens or 2_048, 2_048),
        "context_window": provider.context_window,
    }
    return hashlib.sha256(json.dumps(safe, sort_keys=True).encode()).hexdigest()


def _code_identity(workdir: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(workdir), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    return f"{workdir.resolve()}|{head or 'no-git-head'}"


def _fresh_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = [
        *sorted((root / "src/horizon_harness/workspace").glob("*.py")),
        root / "src/horizon_harness/memory/indexed.py",
        root / "src/horizon_harness/memory/extraction.py",
        root / "runs/task_workspace.py",
        root / "runs/workspace_memory_eval.py",
    ]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(part, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="freeze inputs and matrix without model calls")
    run = sub.add_parser(
        "run", help="execute the frozen paid matrix and official verifier"
    )
    run.add_argument("--dotenv", type=Path, default=ROOT / ".env.history-reuse.glm")
    run.add_argument("--development-attempts", type=int, default=1)
    run.add_argument("--embedding-revision", required=True)
    run.add_argument("--recovery", action="store_true")
    run.add_argument("--workspace-only", action="store_true")
    run.add_argument(
        "--combined-h3",
        action="store_true",
        help="One combined handoff check from the saved H3 prefix",
    )
    run.add_argument(
        "--repair-h3",
        action="store_true",
        help="Reclassify saved reports and retry only H3 under new cell ids",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        plan = build_plan(args.selection, args.output_dir)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    plan_path = args.output_dir / "experiment.json"
    if not plan_path.is_file():
        raise SystemExit("run requires a frozen experiment.json; execute plan first")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if args.repair_h3 or args.combined_h3:
        result_name = (
            "combined-results.json" if args.combined_h3 else "repaired-results.json"
        )
        runner = FormalRunner(plan, args.output_dir, dotenv=args.dotenv)
        prior = json.loads((args.output_dir / "results.json").read_text())
        rows = []
        for original in prior["cells"]:
            row = dict(original)
            if "verifier_result" in row:
                root = Path(row["verifier_result"]).parent
                metadata = json.loads((root / "result.json").read_text())
                row["original_verdict"] = row["verdict"]
                row["verdict"] = _verifier_verdict(
                    row["instance_id"],
                    [Path(p) for p in metadata["reports"]],
                    metadata["exit_code"],
                )
            rows.append(row)
        _write_json(
            args.output_dir / result_name,
            {"cells": rows, "budget_limit": TOTAL_BUDGET},
        )
        _write_json(
            args.output_dir
            / (
                "combined-provenance.json"
                if args.combined_h3
                else "repair-provenance.json"
            ),
            {
                "implementation_fingerprint": _implementation_fingerprint(ROOT),
                "budget_limit": TOTAL_BUDGET,
                "prior_costs_retained": True,
                "experiment": "combined_handoff_one_case"
                if args.combined_h3
                else "h3_repair",
                "note": "H3 context accounting repair; other cells reclassified from existing oracle reports, no new Agent calls.",
            },
        )
        store_path = args.output_dir.resolve() / "snapshots/H3/store"
        record_path = next(store_path.glob("*/task.json"))
        record = WorkspaceStore(store_path).load(record_path.parent.name)
        prefix = {
            "task_id": record.task_id,
            "resumable": True,
            "snapshot": args.output_dir.resolve() / "snapshots/H3",
            "workdir": args.output_dir.resolve() / "runs/prefix-H3/workdir",
            "store": args.output_dir.resolve() / "runs/prefix-H3/store",
        }
        for label, mode in (
            (("C-H3", "combined"),)
            if args.combined_h3
            else (("W-H3-repair", "workspace"), ("S-H3-repair", "summary"))
        ):
            try:
                row = runner._suffix(
                    label,
                    runner.tasks["H3"],
                    prefix,
                    mode=mode,
                    memory_db=args.output_dir.resolve() / "memory/frozen.sqlite3",
                    embedding_revision=args.embedding_revision,
                )
            except Exception as exc:
                row = {
                    "cell": label,
                    "verdict": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            _write_json(
                args.output_dir / result_name,
                {"cells": rows, "budget_limit": TOTAL_BUDGET},
            )
        return 0
    try:
        FormalRunner(plan, args.output_dir, dotenv=args.dotenv).run(
            development_attempts=args.development_attempts,
            embedding_revision=args.embedding_revision,
            recovery=args.recovery,
            workspace_only=args.workspace_only,
        )
    except Exception as exc:
        failure = {"status": "interrupted", "error": f"{type(exc).__name__}: {exc}"}
        _write_json(args.output_dir / "failure.json", failure)
        with (args.output_dir / "report.md").open("a", encoding="utf-8") as stream:
            stream.write("\nInterrupted: " + failure["error"] + "\n")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
