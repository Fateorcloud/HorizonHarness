"""CLI for persistent task start, inspect, and safe resume."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

from horizon_harness.llm.env import (
    FAKE_PROVIDER,
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
from horizon_harness.workspace import TaskSession
from horizon_harness.workspace.drift import drift_report, drifted_step_ids
from horizon_harness.workspace.replay import (
    diff_runs,
    replay,
)
from horizon_harness.workspace.legacy import read_legacy_record, import_legacy_record
from horizon_harness.workspace.verification import AcceptanceContract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a text coding task through durable safe-turn checkpoints. "
            "This is continuation, not arbitrary crash recovery."
        )
    )
    parser.add_argument("--code-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--store-dir", type=Path, default=Path("evals/out/task-workspaces")
    )
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument(
        "--handoff",
        type=Path,
        help="Combined handoff JSON containing text and source_revision",
    )
    parser.add_argument("--memory-db", type=Path)
    parser.add_argument("--memory-scope")
    parser.add_argument(
        "--memory-mode", choices=("keyword", "hybrid"), default="keyword"
    )
    parser.add_argument("--embedding-revision")
    parser.add_argument(
        "--extract-memory-on-complete",
        action="store_true",
        help="run one sourced extraction call only after completed/failed status",
    )
    parser.add_argument(
        "--fake", action="store_true", help="key-free deterministic smoke"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("task")
    start.add_argument("--task-id")
    start.add_argument("--constraint", action="append", default=[])
    start.add_argument("--acceptance", action="append", default=[])
    start.add_argument("--check", action="append", default=[])
    start.add_argument("--contract-id", default="cli-contract-v1")
    start.add_argument("--acceptance-dir", type=Path)
    start.add_argument("--verification-exclude-dir", action="append", default=[])
    _run_args(start)
    resume = sub.add_parser("resume")
    resume.add_argument("task_id")
    _run_args(resume)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("task_id")
    adopt = sub.add_parser("adopt-current")
    adopt.add_argument("task_id")
    adopt.add_argument("--step", required=True)
    adopt.add_argument("--expected-sha", required=True)
    adopt.add_argument("--note", default="")
    replay_cmd = sub.add_parser("replay")
    replay_cmd.add_argument("task_id")
    diff = sub.add_parser("diff")
    diff.add_argument("left_task_id")
    diff.add_argument("right_task_id")
    fork = sub.add_parser("fork")
    fork.add_argument("task_id")
    fork.add_argument("--target-code-dir", type=Path, required=True)
    legacy = sub.add_parser("legacy-inspect")
    legacy.add_argument("path", type=Path)
    legacy_import = sub.add_parser("legacy-import")
    legacy_import.add_argument("path", type=Path)
    legacy_import.add_argument("--task-id")
    verify = sub.add_parser("verify")
    verify.add_argument("task_id")
    verify.add_argument("--check", action="append", required=True)
    verify.add_argument("--contract-id", default="cli-contract-v1")
    complete = sub.add_parser("complete")
    complete.add_argument("task_id")
    complete.add_argument("--check", action="append", required=True)
    complete.add_argument("--contract-id", default="cli-contract-v1")
    return parser


def _run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--pause-after-turns", type=int)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in (
        "inspect",
        "adopt-current",
        "replay",
        "diff",
        "fork",
        "legacy-inspect",
        "legacy-import",
    ):
        if args.command == "legacy-inspect":
            print(
                json.dumps(
                    read_legacy_record(args.path).as_dict(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "legacy-import":
            imported = import_legacy_record(
                args.path,
                store=TaskSession(
                    FAKE_PROVIDER, code_dir=args.code_dir, store_dir=args.store_dir
                ).store,
                task_id=args.task_id,
            )
            print(json.dumps(imported.as_dict(), ensure_ascii=False, indent=2))
            return 0
        session = TaskSession(
            FAKE_PROVIDER,
            code_dir=args.code_dir,
            store_dir=args.store_dir,
        )
        if args.command == "adopt-current":
            try:
                result = session.adopt_current(
                    args.task_id,
                    step_id=args.step,
                    expected_sha=args.expected_sha,
                    note=args.note,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "diff":
            result = diff_runs(
                session.store.journal(args.left_task_id),
                session.store.journal(args.right_task_id),
            )
            print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2))
            return 0
        if args.command == "replay":
            result = replay(session.store.journal(args.task_id))
            print(
                json.dumps(
                    {"events": result.events, "projection": result.projection},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "fork":
            child = session.fork_task(
                args.task_id,
                child_task_id=args.task_id + "-fork",
                child_code_dir=args.target_code_dir,
                child_store_dir=args.store_dir,
            )
            print(
                json.dumps(
                    {
                        "task_id": child.task_id,
                        "status": child.status,
                        "parent_task_id": child.parent_task_id,
                        "parent_revision": child.parent_revision,
                        "workspace_tree_sha256": (
                            child.checkpoint.workspace_tree_sha256
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        inspected = session.inspect(args.task_id).as_dict()
        journal = session.store.journal(args.task_id)
        inspected["drift_reports"] = [
            drift_report(journal, root=session.code_dir, step_id=step_id)
            for step_id in drifted_step_ids(journal)
        ]
        print(json.dumps(inspected, ensure_ascii=False, indent=2))
        return 0
    if args.command in ("verify", "complete"):
        session = TaskSession(
            FAKE_PROVIDER if args.fake else provider_from_env(read_reasoning=True),
            code_dir=args.code_dir,
            store_dir=args.store_dir,
        )
        frozen = session.inspect(args.task_id)
        contract = AcceptanceContract(
            args.contract_id,
            tuple(args.check),
            acceptance_dir=frozen.acceptance_dir or None,
            acceptance_tree_sha256=frozen.acceptance_tree_sha256,
            verification_exclude_dirs=frozen.verification_exclude_dirs or (),
        )
        if args.command == "verify":
            receipt = session.verify(args.task_id, contract)
            print(json.dumps(dataclasses.asdict(receipt), ensure_ascii=False, indent=2))
        else:
            record = session.complete_if_verified(args.task_id, contract)
            print(json.dumps(record.as_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.fake:
        provider = FAKE_PROVIDER
        request_extra = None
    else:
        load_dotenv(args.dotenv)
        provider = provider_from_env(read_reasoning=True)
        request_extra = request_extra_from_env(env=os.environ)
    embedder = None
    if args.memory_mode == "hybrid":
        if not args.embedding_revision:
            raise SystemExit("--embedding-revision is required for hybrid memory")
        embedder = SentenceTransformerEmbedder(
            revision=args.embedding_revision,
        )
    memory = (
        IndexedMemory(args.memory_db, mode=args.memory_mode, embedder=embedder)
        if args.memory_db
        else None
    )
    if args.extract_memory_on_complete and memory is None:
        raise SystemExit("--extract-memory-on-complete requires --memory-db")
    if args.extract_memory_on_complete and args.fake:
        raise SystemExit("real memory extraction cannot use the fake provider")
    extractor = (
        ModelMemoryExtractor(provider, request_extra=request_extra)
        if args.extract_memory_on_complete
        else None
    )
    spec = AgentSpec(
        name="workspace-agent",
        role="Complete the repository task with the available bash, read, and edit tools.",
        capabilities=("read", "edit"),
        workflow="single",
    )
    session = TaskSession(
        provider,
        spec=spec,
        code_dir=args.code_dir,
        store_dir=args.store_dir,
        request_extra=request_extra,
        memory=memory,
        memory_scope=args.memory_scope or str(args.code_dir.resolve()),
        memory_extractor=extractor,
        context_representation="combined" if args.handoff else "workspace",
        continuation_summary=(
            json.loads(args.handoff.read_text(encoding="utf-8"))["text"]
            if args.handoff
            else ""
        ),
        summary_revision=(
            json.loads(args.handoff.read_text(encoding="utf-8"))["source_revision"]
            if args.handoff
            else None
        ),
    )
    if args.command == "start":
        if (args.acceptance_dir or args.verification_exclude_dir) and not args.check:
            raise SystemExit(
                "--acceptance-dir and --verification-exclude-dir require --check"
            )
        result = session.start(
            args.task,
            constraints=args.constraint,
            acceptance=args.acceptance,
            acceptance_contract=(
                AcceptanceContract(
                    args.contract_id,
                    tuple(args.check),
                    acceptance_dir=args.acceptance_dir,
                    verification_exclude_dirs=tuple(args.verification_exclude_dir),
                )
                if args.check
                else None
            ),
            task_id=args.task_id,
            max_turns=args.max_turns,
            pause_after_turns=args.pause_after_turns,
        )
    else:
        result = session.resume(
            args.task_id,
            max_turns=args.max_turns,
            pause_after_turns=args.pause_after_turns,
        )
    print(
        json.dumps(
            {
                "task_id": result.task_id,
                "run_id": result.run_id,
                "status": result.status,
                "revision": result.record.revision,
                "checkpoint": result.record.checkpoint.messages_path,
                "trace": result.record.checkpoint.trace_path,
                "memory_entry_ids": result.memory_entry_ids,
                "memory_extraction_error": result.memory_extraction_error,
                "memory_extraction_usage": (
                    dataclasses.asdict(result.memory_extraction.usage)
                    if result.memory_extraction is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
