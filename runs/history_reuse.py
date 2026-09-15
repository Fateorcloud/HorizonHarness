"""Isolated, explicitly bounded history-reuse experiments (not a benchmark)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import dataclasses
import hashlib
import subprocess
import tempfile

from openai import OpenAI, APIStatusError

from horizon_harness.llm.env import load_dotenv
from horizon_harness import AgentSession, AgentSpec, Provider, run
from horizon_harness.compression import summarize_compression
from horizon_harness.tools import make_recall_tool
from horizon_harness.trace import run_trace_from_state, write_event_stream
from runs.history_reuse_cases import CASES, HOLDOUT_CASES, seed_case
from runs.history_reuse_model import BudgetExhausted, Ledger, bounded_generate
from runs.history_reuse_policy import policy_for


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def pilot(
    dotenv: Path, output: Path, groups: str = "ABC", split: str = "debug"
) -> None:
    """Run the frozen six-cell constructed pilot, never overwrite earlier results."""
    env: dict[str, str] = {}
    load_dotenv(dotenv, environ=env)
    token = env.get("OPENAI_AUTH_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing experiment key")
    output.mkdir(parents=True, exist_ok=False)
    cases = HOLDOUT_CASES if split == "holdout" else CASES
    provider = Provider(
        id="history-reuse",
        api="openai-chat",
        model=env["OPENAI_MODEL"],
        base_url=env["OPENAI_BASE_URL"],
        default_temperature=None,
    )
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("history_reuse_cases.py"),
        Path(__file__).with_name("history_reuse_model.py"),
        Path(__file__).with_name("history_reuse_policy.py"),
    ]
    manifest = {
        "kind": "constructed-continuation-pilot",
        "split": split,
        "selected_groups": groups,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
        "provider": dataclasses.asdict(provider),
        "groups": {
            "A": "no compression, no recall",
            "B": "tool-compact, no recall",
            "C": "tool-compact, existing recall",
            "D": "tool-compact with source indices, existing recall",
        },
        "strategy": {
            "threshold_tokens": 4000,
            "keep_recent_exchanges": 1,
            "preview_chars": 200,
        },
        "per_run_limits": {
            "requests": 12,
            "input_tokens": 64000,
            "output_tokens": 8000,
            "seconds": 600,
            "sdk_retries": 0,
            "per_call_output": 2048,
        },
        "input_guard": "UTF-8 wire bytes + 4096 proxy before call; actual usage after call; not exact tokenizer",
        "cases": [dataclasses.asdict(case) for case in cases],
        "price": None,
        "tool_schemas": {},
    }
    write_json(output / "manifest.json", manifest)
    results = []
    for case in cases:
        for group in groups:
            cell = output / f"{case.name}-{group}"
            cell.mkdir()
            ledger = Ledger()
            status = "completed"
            with tempfile.TemporaryDirectory(prefix="horizon-history-") as tmp:
                workspace = Path(tmp)
                state = seed_case(case, workspace)
                initial = {p.name: p.read_text() for p in workspace.iterdir()}
                write_json(cell / "initial-workspace.json", initial)
                policy = policy_for(group)
                recall = (make_recall_tool(state),) if group in "CD" else ()
                with (
                    AgentSession(
                        provider,
                        spec=AgentSpec(
                            name="solver",
                            capabilities=("bash", "read", "edit"),
                            system_prompt="Complete the coding task in the current workspace. Use available tools as needed. Verify your implementation.",
                        ),
                        cwd=workspace,
                        tools=recall,
                        context_policy=policy,
                    ) as session,
                    OpenAI(
                        api_key=token,
                        base_url=provider.base_url,
                        max_retries=0,
                        timeout=60,
                    ) as client,
                ):
                    manifest["tool_schemas"][group] = [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                        for t in session.agent.tools
                    ]
                    write_json(output / "manifest.json", manifest)
                    agent = dataclasses.replace(
                        session.agent,
                        generate=bounded_generate(session.agent, client, ledger),
                    )
                    try:
                        list(
                            run(
                                agent,
                                state,
                                max_turns=12,
                                abort=lambda: time.monotonic() - ledger.started >= 600,
                            )
                        )
                    except BudgetExhausted as exc:
                        status = str(exc)
                    except Exception as exc:
                        status = "error:" + type(exc).__name__
                    try:
                        grade = subprocess.run(
                            ["uv", "run", "--no-project", "python", "-c", case.checks],
                            cwd=workspace,
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        passed = grade.returncode == 0
                        (cell / "grade.txt").write_text(
                            grade.stdout + grade.stderr, encoding="utf-8"
                        )
                    except subprocess.TimeoutExpired:
                        passed = False
                    (cell / "solution.py").write_text(
                        (workspace / "solution.py").read_text(), encoding="utf-8"
                    )
                write_event_stream(
                    cell / "trajectory.jsonl",
                    run_trace_from_state(
                        state=state,
                        trace_id=cell.name,
                        producer="history-reuse-pilot",
                    ),
                )
                write_json(cell / "wire.json", ledger.requests)
                result = {
                    "case": case.name,
                    "group": group,
                    "passed": passed,
                    "status": status,
                    "calls": ledger.calls,
                    "input_tokens": ledger.input_tokens,
                    "output_tokens": ledger.output_tokens,
                    "cost": None,
                    "elapsed_seconds": time.monotonic() - ledger.started,
                    "compression": summarize_compression(state.events).as_dict(),
                }
                results.append(result)
                write_json(output / "results.json", results)
                print(json.dumps(result), flush=True)
            if status.startswith("error:"):
                return  # Infrastructure failure is not a reason to spend on more cells.


def preflight(dotenv: Path, output: Path) -> bool:
    """Make one small request, with SDK retries disabled and no ambient creds."""
    env: dict[str, str] = {}
    load_dotenv(dotenv, environ=env)
    token = env.get("OPENAI_AUTH_TOKEN", "").strip()
    if not token:
        raise SystemExit(f"Fill OPENAI_AUTH_TOKEN in {dotenv}")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "model": env["OPENAI_MODEL"],
        "base_url": env["OPENAI_BASE_URL"],
        "requests": 1,
        "max_output_tokens": 64,
        "timeout_seconds": 45,
        "sdk_retries": 0,
        "cost": None,
    }
    started = time.monotonic()
    try:
        with OpenAI(
            api_key=token,
            base_url=env["OPENAI_BASE_URL"],
            max_retries=0,
            timeout=45,
        ) as client:
            response = client.chat.completions.create(
                model=env["OPENAI_MODEL"],
                messages=[{"role": "user", "content": "Reply only OK."}],
                max_tokens=64,
            )
        record.update(
            status="ok",
            response_model=response.model,
            usage=response.usage.model_dump() if response.usage else None,
            finish_reason=response.choices[0].finish_reason
            if response.choices
            else None,
        )
    except APIStatusError as exc:
        record.update(
            status="api_error",
            http_status=exc.status_code,
            error=str(exc).replace(token, "[REDACTED]"),
        )
    except Exception as exc:
        # Network errors may contain endpoint details; never persist credentials.
        record.update(status="connection_error", error_type=type(exc).__name__)
    record["elapsed_seconds"] = time.monotonic() - started
    rendered = json.dumps(record, ensure_ascii=False, indent=2)
    (output / "preflight.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)
    return record["status"] == "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--groups", choices=("ABC", "D", "ABCD"), default="ABC")
    parser.add_argument("--split", choices=("debug", "holdout"), default="debug")
    args = parser.parse_args()
    if args.pilot:
        pilot(args.dotenv, args.output, args.groups, args.split)
        return
    raise SystemExit(0 if preflight(args.dotenv, args.output) else 1)


if __name__ == "__main__":
    main()
