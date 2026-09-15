"""Key-free durable decision, crash, recovery, verification, and replay demo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from horizon_harness.llm.env import FAKE_PROVIDER
from horizon_harness.llm.types import LLMResponse
from horizon_harness.messages import TextBlock, ToolCallBlock
from horizon_harness.workspace import TaskSession
from horizon_harness.workspace.replay import replay
from horizon_harness.workspace.verification import AcceptanceContract


def _decision() -> LLMResponse:
    return LLMResponse(
        content=tuple(
            ToolCallBlock(
                call_id,
                "edit",
                {
                    "path": f"{call_id.lower()}.txt",
                    "old_string": "0",
                    "new_string": value,
                },
            )
            for call_id, value in (("A", "1"), ("B", "2"), ("C", "3"))
        ),
        stop_reason="tool_use",
        model="fake-model",
    )


def _crashing_start(code_dir: Path, store_dir: Path) -> None:
    from horizon_harness.workspace.journal import ExecutionJournal

    commit = ExecutionJournal.commit_model_decision

    def commit_then_exit(self, **kwargs):
        result = commit(self, **kwargs)
        if kwargs["batch_id"]:
            os._exit(91)
        return result

    ExecutionJournal.commit_model_decision = commit_then_exit
    contract = AcceptanceContract("demo-contract-v1", ("true",))
    TaskSession(
        FAKE_PROVIDER,
        code_dir=code_dir,
        store_dir=store_dir,
        complete_fn=lambda request: _decision(),
    ).start(
        "Apply the durable A/B/C edits.",
        task_id="durable-demo",
        acceptance_contract=contract,
    )


def run_demo(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    code_dir = output / "code"
    store_dir = output / "store"
    code_dir.mkdir()
    for name in ("a", "b", "c"):
        (code_dir / f"{name}.txt").write_text("0", encoding="utf-8")
    crashed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--code-dir",
            str(code_dir),
            "--store-dir",
            str(store_dir),
        ],
        check=False,
    )
    if crashed.returncode != 91:
        raise RuntimeError(
            f"fault injection returned {crashed.returncode}, expected 91"
        )

    model_requests = []

    def finish(request):
        model_requests.append(request)
        assistant = [message for message in request.messages if message.tool_calls]
        results = [message for message in request.messages if message.tool_results]
        if len(assistant) != 1 or len(results) != 1:
            raise RuntimeError("recovered provider history is not paired exactly once")
        if tuple(call.id for call in assistant[0].tool_calls) != ("A", "B", "C"):
            raise RuntimeError("recovered AssistantMessage order changed")
        if tuple(item.tool_call_id for item in results[0].tool_results) != (
            "A",
            "B",
            "C",
        ):
            raise RuntimeError("recovered ToolResult order changed")
        return LLMResponse(
            content=(TextBlock("recovered"),),
            stop_reason="end_turn",
            model="fake-model",
        )

    session = TaskSession(
        FAKE_PROVIDER,
        code_dir=code_dir,
        store_dir=store_dir,
        complete_fn=finish,
    )
    session.resume("durable-demo", pause_after_turns=1)
    contract = AcceptanceContract("demo-contract-v1", ("true",))
    receipt = session.verify("durable-demo", contract)
    completed = session.complete_if_verified("durable-demo", contract)
    journal = session.store.journal("durable-demo")
    replayed = replay(journal)
    return {
        "output": str(output),
        "crash_exit_code": crashed.returncode,
        "files": {
            name: (code_dir / f"{name}.txt").read_text(encoding="utf-8")
            for name in ("a", "b", "c")
        },
        "next_model_calls": len(model_requests),
        "verification_outcome": receipt.outcome,
        "status": completed.status,
        "journal_event_count": len(replayed.events),
        "bundle_commit_count": sum(
            kind == "ToolBatchBundleCommitted" for _, kind, _ in replayed.events
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        if args.code_dir is None or args.store_dir is None:
            parser.error("--child requires --code-dir and --store-dir")
        _crashing_start(args.code_dir, args.store_dir)
        return 0
    output = args.output or Path("evals/out") / f"durable-demo-{uuid.uuid4().hex[:8]}"
    print(json.dumps(run_demo(output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
