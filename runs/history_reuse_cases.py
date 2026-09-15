"""Constructed continuation cases, not historical benchmark failures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from horizon_harness import State
from horizon_harness.messages import (
    ToolCallBlock,
    assistant_message,
    tool_result_message,
)


@dataclass(frozen=True)
class Case:
    name: str
    task: str
    fact: str
    initial_code: str
    checks: str
    recoverable_from_file: bool
    fact_exchange: int = 0


CASES = (
    Case(
        "file-contract",
        "Finish implementing normalize_event(value) in solution.py according to the "
        "interface contract inspected earlier. Run checks and leave the code ready. "
        "Work only in the current workspace. The earlier exploration is already done.",
        "Interface contract: normalize_event(value) strips leading/trailing whitespace, "
        "lowercases ASCII letters, replaces each hyphen with an underscore WITHOUT "
        "collapsing repeated separators, and prefixes the result with 'evt.q7/'.",
        "def normalize_event(value):\n    return value\n",
        "from solution import normalize_event\n"
        "assert normalize_event('  A--B  ') == 'evt.q7/a__b'\n"
        "assert normalize_event(' X_Y ') == 'evt.q7/x_y'\n"
        "assert normalize_event('') == 'evt.q7/'\n",
        True,
    ),
    Case(
        "historical-receipt",
        "Finish implementing audit_key(sequence) in solution.py using the run namespace "
        "from the earlier transient test receipt. Return namespace + ':' + the sequence "
        "as six decimal digits padded with zeros. The receipt is a historical observation, "
        "not a current workspace file. Work only in the current workspace.",
        "Transient test receipt: this run's audit namespace is 'canary-m8v2'. "
        "audit_key(17) must be 'canary-m8v2:000017'.",
        "def audit_key(sequence):\n    return str(sequence)\n",
        "from solution import audit_key\n"
        "assert audit_key(17) == 'canary-m8v2:000017'\n"
        "assert audit_key(0) == 'canary-m8v2:000000'\n"
        "assert audit_key(123456) == 'canary-m8v2:123456'\n",
        False,
    ),
)


HOLDOUT_CASES = (
    Case(
        "retry-classification",
        "Implement should_retry(status, attempt) in solution.py according to the "
        "service policy inspected earlier. attempt counts completed retries, starting "
        "at zero. Use the current workspace only and verify the implementation.",
        "Service policy: retry only HTTP 409, 425, and 503, and only while attempt < 2. "
        "Do not retry other statuses, including 429, 500, and 502. Negative attempts "
        "must raise ValueError even for a status that is not retryable.",
        "def should_retry(status, attempt):\n    return status >= 500\n",
        "from solution import should_retry\n"
        "for status in [200, 400, 409, 425, 429, 500, 502, 503, 504]:\n"
        "    for attempt in range(4):\n"
        "        assert should_retry(status, attempt) == (status in {409, 425, 503} and attempt < 2)\n"
        "for status in [200, 503]:\n"
        "    try:\n        should_retry(status, -1)\n"
        "    except ValueError:\n        pass\n"
        "    else:\n        raise AssertionError('negative attempt accepted')\n",
        True,
        2,
    ),
    Case(
        "legacy-wire-parser",
        "Implement parse_record(line) in solution.py using the legacy wire format "
        "established by the earlier compatibility probe. Return (name, count), with "
        "count an integer; malformed records must raise ValueError. The earlier "
        "probe output is historical, not a workspace file. Work only in this workspace.",
        "Compatibility probe result: a record is COUNT~NAME, split on the FIRST '~'. "
        "COUNT is one or more ASCII digits, no signs or whitespace. NAME is nonempty "
        "and preserved verbatim, including additional '~' characters and spaces. "
        "For example '007~north~west' maps to ('north~west', 7).",
        "def parse_record(line):\n    return line, 0\n",
        "from solution import parse_record\n"
        "assert parse_record('007~north~west') == ('north~west', 7)\n"
        "assert parse_record('0~ x ') == (' x ', 0)\n"
        "assert parse_record('42~ok') == ('ok', 42)\n"
        "for line in ['', 'abc', '1~', '~name', '-1~x', '+1~x', ' 1~x', '1 ~x', '١~x']:\n"
        "    try:\n        parse_record(line)\n"
        "    except ValueError:\n        pass\n"
        "    else:\n        raise AssertionError(repr(line))\n",
        False,
        4,
    ),
)


def seed_case(case: Case, workspace: Path) -> State:
    """Seed paired history; fact message index is 2 * fact_exchange + 2."""
    (workspace / "solution.py").write_text(case.initial_code, encoding="utf-8")
    if case.recoverable_from_file:
        (workspace / "CONTRACT.txt").write_text(case.fact + "\n", encoding="utf-8")
    state = State(case.task)
    state.send("task", "user", "solver", case.task)
    for index in range(6):
        call_id = f"seed-{index}"
        if index == case.fact_exchange:
            command = (
                "cat CONTRACT.txt"
                if case.recoverable_from_file
                else "transient-test --receipt"
            )
            body = "Earlier inspection output:\n" + "diagnostic ok\n" * 35 + case.fact
        else:
            command = f"inspect-module --part {index}"
            body = f"Exploration part {index}: no further interface constraints.\n"
        body += "\n" + "ordinary diagnostic filler\n" * 105
        state.record(
            assistant_message(
                [ToolCallBlock(call_id, "bash", {"command": command})],
                sender="solver",
                target="user",
                kind="step",
            )
        )
        state.record(
            tool_result_message(
                body,
                tool_call_id=call_id,
                tool_name="bash",
                target="solver",
            )
        )
    return state
