"""Container-local runner for the HorizonHarness Harbor agent."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

from horizon_harness.agents.starter import AgentSession
from horizon_harness.evals.harbor import DEFAULT_API_KIND, DEFAULT_MAX_TURNS
from horizon_harness.llm import Provider
from horizon_harness.llm.env import (
    API_KIND_CHOICES,
    API_KIND_ENV,
    FAKE_PROVIDER,
    OPENAI_ENV,
    provider_from_env as _provider_from_env,
    request_extra_from_env,
)
from horizon_harness.messages import AssistantMessage, message_text
from horizon_harness.state import State
from horizon_harness.spec import (
    AgentSpec,
    CAPABILITIES,
    Capability,
    Workflow,
    WORKFLOWS,
)
from horizon_harness.trace import run_trace_from_state, write_event_stream


DEFAULT_TRACE_PATH = "/logs/agent/horizon-trajectory.jsonl"
DEFAULT_SUMMARY_PATH = "/logs/agent/horizon-summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--capabilities",
        default="bash,read,task",
        help=f"Comma-separated capabilities chosen from {CAPABILITIES}.",
    )
    parser.add_argument("--workflow", choices=WORKFLOWS, default="single")
    parser.add_argument("--provider", choices=["openai", "fake"], default="openai")
    parser.add_argument(
        "--api-kind",
        choices=API_KIND_CHOICES,
        # env-ok: CLI default mirrors API_KIND for the Harbor in-container entrypoint
        default=os.environ.get(API_KIND_ENV, DEFAULT_API_KIND),
    )
    parser.add_argument("--trace-id", default="harbor.horizon-harness")
    parser.add_argument("--producer", default="harbor:horizon-harness")
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def load_instruction(args: argparse.Namespace) -> str:
    if args.instruction_file:
        return Path(args.instruction_file).read_text(encoding="utf-8")
    if args.instruction:
        return args.instruction
    raise SystemExit("missing instruction: pass --instruction or --instruction-file")


def provider_from_runner_args(args: argparse.Namespace) -> Provider:
    if args.provider == "fake":
        return FAKE_PROVIDER
    provider = _provider_from_env(
        OPENAI_ENV,
        api_kind=args.api_kind,
        default_temperature=1.0,
        read_reasoning=True,
        label="the Harbor HorizonHarness runner provider",
    )
    if args.max_output_tokens is None:
        return provider
    if args.max_output_tokens < 1:
        raise SystemExit("--max-output-tokens must be at least 1")
    return dataclasses.replace(
        provider,
        default_max_tokens=args.max_output_tokens,
    )


def build_runner_session(
    *,
    provider: Provider,
    cwd: str | Path,
    capabilities: str,
    workflow: str,
    max_turns: int,
    request_extra: dict[str, Any] | None = None,
) -> AgentSession:
    selected = cast(
        tuple[Capability, ...],
        tuple(part.strip() for part in capabilities.split(",") if part.strip()),
    )
    selected_workflow = cast(Workflow, workflow)
    return AgentSession(
        provider,
        spec=AgentSpec(capabilities=selected, workflow=selected_workflow),
        cwd=cwd,
        request_extra=request_extra,
        max_turns=max_turns,
    )


def final_text_from_state(state: State) -> str:
    for message in reversed(state.messages):
        if isinstance(message, AssistantMessage):
            text = message_text(message)
            if text:
                return text
    return ""


def _write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    instruction = load_instruction(args)
    provider = provider_from_runner_args(args)
    started = time.monotonic()
    request_extra = request_extra_from_env() if args.provider == "openai" else {}

    with build_runner_session(
        provider=provider,
        cwd=args.cwd,
        capabilities=args.capabilities,
        workflow=args.workflow,
        max_turns=args.max_turns,
        request_extra=request_extra,
    ) as session:
        state, events = session.run(instruction, max_turns=args.max_turns)
        for _event in events:
            pass
        try:
            trace_state = session.agent.trace_state(state)
        except Exception:
            trace_state = state

    trace = run_trace_from_state(
        state=trace_state,
        trace_id=args.trace_id,
        producer=args.producer,
        meta={
            "harbor": True,
            "capabilities": args.capabilities.split(","),
            "workflow": args.workflow,
            "provider": args.provider,
            "cwd": str(args.cwd),
        },
    )
    write_event_stream(args.trace_path, trace)

    summary = {
        "status": "ok",
        "provider": args.provider,
        "model": provider.model,
        "capabilities": args.capabilities.split(","),
        "workflow": args.workflow,
        "max_turns": args.max_turns,
        "max_output_tokens": provider.default_max_tokens,
        "cwd": str(args.cwd),
        "trace_path": str(args.trace_path),
        "final": final_text_from_state(state),
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    _write_summary(args.summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_once(args)
    except BaseException as exc:
        summary = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "provider": getattr(args, "provider", None),
            "capabilities": getattr(args, "capabilities", None),
            "workflow": getattr(args, "workflow", None),
        }
        try:
            _write_summary(args.summary_path, summary)
        except Exception:
            pass
        print(
            f"[horizon-harness] runner failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
