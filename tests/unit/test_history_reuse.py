"""Offline mechanism checks; none claim real-model task effectiveness."""

import tempfile
import unittest
import os
from pathlib import Path

from horizon_harness import AgentSession, AgentSpec, Provider
from horizon_harness.compression import ToolCompactStrategy, maybe_compress_context
from horizon_harness.context_view import ContextPolicy
from horizon_harness.messages import text_of, tool_results_of
from horizon_harness.tools import make_recall_tool, tool_result_text
from runs.history_reuse_cases import CASES, HOLDOUT_CASES, seed_case
from runs.history_reuse_model import BudgetExhausted, Ledger, bounded_generate
from runs.history_reuse_policy import policy_for


class HistoryReuseTest(unittest.TestCase):
    def test_timeout_is_forwarded_and_error_is_recorded(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from horizon_harness import Agent

        create = Mock(side_effect=TimeoutError("not persisted"))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        ledger = Ledger(request_timeout_seconds=180)
        agent = Agent(
            "a",
            lambda _: None,
            llm_provider=Provider(id="test", api="openai-chat", model="test"),
        )
        with self.assertRaises(TimeoutError):
            bounded_generate(agent, client, ledger)([])
        self.assertEqual(create.call_count, 1)
        self.assertEqual(create.call_args.kwargs["timeout"], 180)
        self.assertEqual(ledger.requests[0]["error"]["type"], "TimeoutError")
        self.assertIsNone(ledger.requests[0]["error"]["http_status"])
        self.assertIn("elapsed_seconds", ledger.requests[0])

    def test_larger_input_budget_preserves_other_limits(self):
        ledger = Ledger(input_tokens=64000, max_input_tokens=192000)
        self.assertEqual(ledger.reserve({}), 2048)
        ledger.input_tokens = 190000
        with self.assertRaises(BudgetExhausted):
            ledger.reserve({})
        ledger.input_tokens = 0
        ledger.calls = 12
        with self.assertRaises(BudgetExhausted):
            ledger.reserve({})

    def test_holdouts_have_distinct_tasks_and_fact_indices(self):
        self.assertFalse({c.name for c in CASES} & {c.name for c in HOLDOUT_CASES})
        for case in HOLDOUT_CASES:
            with self.subTest(case=case.name), tempfile.TemporaryDirectory() as tmp:
                state = seed_case(case, Path(tmp))
                expected = 2 * case.fact_exchange + 2
                self.assertIn(
                    case.fact,
                    text_of(
                        tool_results_of(state.messages[expected].content)[0].content
                    ),
                )
                baseline = policy_for("C").strategy(
                    state.active_context_items(), "solver"
                )
                treatment = policy_for("D").strategy(
                    state.active_context_items(), "solver"
                )
                self.assertIn(expected, baseline.compress_indices)
                self.assertNotIn(case.fact, text_of(treatment.replacement.content))
                self.assertIn(
                    f"original transcript messages {expected}",
                    text_of(treatment.replacement.content),
                )

    def test_holdout_graders_reject_stubs_accept_references(self):
        import subprocess

        references = (
            "def should_retry(status, attempt):\n"
            "    if attempt < 0:\n        raise ValueError(attempt)\n"
            "    return status in {409, 425, 503} and attempt < 2\n",
            "def parse_record(line):\n"
            "    count, sep, name = line.partition('~')\n"
            "    if not sep or not name or not count or any(c not in '0123456789' for c in count):\n"
            "        raise ValueError(line)\n"
            "    return name, int(count)\n",
        )
        for case, reference in zip(HOLDOUT_CASES, references, strict=True):
            with self.subTest(case=case.name), tempfile.TemporaryDirectory() as tmp:
                seed_case(case, Path(tmp))
                command = [
                    os.environ.get("UV", "uv"),
                    "run",
                    "--no-project",
                    "python",
                    "-B",
                    "-c",
                    case.checks,
                ]
                bad = subprocess.run(command, cwd=tmp, capture_output=True, timeout=15)
                self.assertNotEqual(bad.returncode, 0)
                (Path(tmp) / "solution.py").write_text(reference)
                good = subprocess.run(command, cwd=tmp, capture_output=True, timeout=15)
                self.assertEqual(good.returncode, 0, good.stderr)

    def test_baseline_policy_removes_only_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = seed_case(CASES[0], Path(tmp))
            active = state.active_context_items()
            baseline = policy_for("C").strategy(active, "solver")
            treatment = policy_for("D").strategy(active, "solver")
            self.assertEqual(baseline.compress_indices, treatment.compress_indices)
            self.assertNotIn(
                "original transcript", text_of(baseline.replacement.content)
            )
            self.assertIn(
                "original transcript messages 2", text_of(treatment.replacement.content)
            )
            self.assertTrue(
                text_of(baseline.replacement.content).startswith(
                    "Compacted 5 older tool exchange(s):\n- bash: 'Earlier inspection output:"
                )
            )

    def test_external_assertions_reject_stubs_and_accept_reference(self):
        import subprocess

        implementations = (
            "def normalize_event(value):\n"
            "    import string\n"
            "    table = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)\n"
            "    return 'evt.q7/' + value.strip().translate(table).replace('-', '_')\n",
            "def audit_key(sequence):\n    return f'canary-m8v2:{sequence:06d}'\n",
        )
        for case, reference in zip(CASES, implementations, strict=True):
            with self.subTest(case=case.name), tempfile.TemporaryDirectory() as tmp:
                seed_case(case, Path(tmp))
                command = [
                    os.environ.get("UV", "uv"),
                    "run",
                    "--no-project",
                    "python",
                    "-B",
                    "-c",
                    case.checks,
                ]
                bad = subprocess.run(command, cwd=tmp, capture_output=True, timeout=15)
                self.assertNotEqual(bad.returncode, 0)
                (Path(tmp) / "solution.py").write_text(reference)
                good = subprocess.run(command, cwd=tmp, capture_output=True, timeout=15)
                self.assertEqual(good.returncode, 0, good.stderr)

    def test_constructed_fact_visibility_and_retrieval(self):
        for case in CASES:
            with self.subTest(case=case.name), tempfile.TemporaryDirectory() as tmp:
                state = seed_case(case, Path(tmp))
                original = state.messages
                self.assertIn(
                    case.fact, text_of(tool_results_of(original[2].content)[0].content)
                )
                self.assertEqual(
                    (Path(tmp) / "CONTRACT.txt").exists(), case.recoverable_from_file
                )
                tool = make_recall_tool(state)
                with AgentSession(
                    Provider(id="fake", api="fake", model="fake"),
                    spec=AgentSpec(
                        name="solver", capabilities=("bash", "read", "edit")
                    ),
                    tools=(tool,),
                    cwd=tmp,
                ) as session:
                    self.assertEqual(
                        [t.name for t in session.agent.tools],
                        ["bash", "read", "edit", "recall"],
                    )
                    events = maybe_compress_context(
                        session.agent,
                        state,
                        ContextPolicy(strategy=ToolCompactStrategy(4000)),
                    )
                self.assertTrue(events)
                self.assertNotIn(2, [i for i, _ in state.active_context_items()])
                self.assertEqual(state.messages[: len(original)], original)
                visible = "\n".join(
                    text_of(m.content) for m in state.active_context_messages()
                )
                self.assertNotIn(case.fact, visible)
                self.assertIn("original transcript messages 2", visible)
                before = list(state.events)
                result = tool.execute("test", {"indices": [2]}, lambda: False, None)
                self.assertIn(case.fact, tool_result_text(result))
                self.assertEqual(state.events, before)

    def test_budget_stops_before_extra_request(self):
        for ledger in (
            Ledger(calls=12),
            Ledger(input_tokens=63000),
            Ledger(output_tokens=8000),
        ):
            with self.subTest(ledger=ledger), self.assertRaises(BudgetExhausted):
                ledger.reserve({})
        ledger = Ledger(output_tokens=7900)
        self.assertEqual(ledger.reserve({}), 100)
        self.assertEqual(ledger.calls, 1)


if __name__ == "__main__":
    unittest.main()
