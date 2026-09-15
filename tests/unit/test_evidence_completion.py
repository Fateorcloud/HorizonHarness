import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from horizon_harness.workspace.verification import (
    AcceptanceContract,
    receipt_is_current,
    run_verifier,
)
from horizon_harness.llm.env import FAKE_PROVIDER
from horizon_harness.workspace import TaskSession
from horizon_harness.workspace.store import UnsafeCheckpointError


class EvidenceCompletionTests(unittest.TestCase):
    def test_task_state_change_invalidates_verification_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            contract = AcceptanceContract("frozen", ("true",))
            session = TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=Path(tmp) / "store"
            )
            task = session.start(
                "verify", acceptance_contract=contract, pause_after_turns=1
            )
            session.verify(task.task_id, contract)
            current = session.inspect(task.task_id)
            session.store.save(
                replace(
                    current,
                    revision=current.revision + 1,
                    next_action="changed after verification",
                )
            )

            with self.assertRaisesRegex(UnsafeCheckpointError, "stale"):
                session.complete_if_verified(task.task_id, contract)

    def test_completed_task_still_rejects_stale_code_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            checked = root / "checked.txt"
            checked.write_text("accepted", encoding="utf-8")
            contract = AcceptanceContract(
                "frozen", ('test "$(cat checked.txt)" = accepted',)
            )
            session = TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=Path(tmp) / "store"
            )
            task = session.start(
                "verify", acceptance_contract=contract, pause_after_turns=1
            )
            session.verify(task.task_id, contract)
            session.complete_if_verified(task.task_id, contract)
            checked.write_text("changed", encoding="utf-8")

            with self.assertRaisesRegex(UnsafeCheckpointError, "stale"):
                session.complete_if_verified(task.task_id, contract)

    def test_task_freezes_acceptance_contract_at_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            session = TaskSession(
                FAKE_PROVIDER, code_dir=root, store_dir=Path(tmp) / "store"
            )
            frozen = AcceptanceContract("frozen", ("true",))
            task = session.start(
                "verify", acceptance_contract=frozen, pause_after_turns=1
            )
            self.assertEqual(task.record.acceptance_contract_id, "frozen")

            with self.assertRaisesRegex(UnsafeCheckpointError, "frozen"):
                session.verify(
                    task.task_id, AcceptanceContract("replacement", ("true",))
                )

    def test_successful_receipt_does_not_override_unsettled_execution(self):
        for pending_state in ("uncommitted_turn", "planned", "ambiguous"):
            with (
                self.subTest(state=pending_state),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp) / "code"
                root.mkdir()
                session = TaskSession(
                    FAKE_PROVIDER, code_dir=root, store_dir=Path(tmp) / "store"
                )
                contract = AcceptanceContract("c", ("true",))
                task = session.start(
                    "verify", acceptance_contract=contract, pause_after_turns=1
                )
                session.verify(task.task_id, contract)
                if pending_state == "uncommitted_turn":
                    session.store.begin_turn(task.task_id, "interrupted", 2)
                else:
                    journal = session.store.journal(task.task_id)
                    journal.plan_step(
                        step_id="pending",
                        tool_name="edit",
                        input_hash="h",
                        idempotency_key="k",
                        effect_class="idempotent_effect",
                    )
                    if pending_state == "ambiguous":
                        journal.set_step_state("pending", "ambiguous", {})
                with self.assertRaisesRegex(
                    UnsafeCheckpointError, "uncommitted|unsettled"
                ):
                    session.complete_if_verified(task.task_id, contract)
                self.assertEqual(session.inspect(task.task_id).status, "paused")

    def test_task_session_persists_verification_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            (root / "check.py").write_text("print('ok')", encoding="utf-8")
            store = Path(tmp) / "store"
            session = TaskSession(FAKE_PROVIDER, code_dir=root, store_dir=store)
            contract = AcceptanceContract("c", ("python check.py",))
            task = session.start(
                "verify", acceptance_contract=contract, pause_after_turns=1
            )
            receipt = session.verify(task.task_id, contract)
            self.assertEqual(
                session.inspect(task.task_id).verifier_results[-1].tree_sha256,
                receipt.tree_sha256,
            )
            completed = session.complete_if_verified(task.task_id, contract)
            self.assertEqual(completed.status, "completed")

    def test_completion_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            store = Path(tmp) / "store"
            session = TaskSession(FAKE_PROVIDER, code_dir=root, store_dir=store)
            contract = AcceptanceContract("c", ("true",))
            task = session.start(
                "verify", acceptance_contract=contract, pause_after_turns=1
            )
            with self.assertRaises(UnsafeCheckpointError):
                session.complete_if_verified(task.task_id, contract)

    def test_receipt_is_bound_to_current_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "check.py").write_text("print('ok')", encoding="utf-8")
            contract = AcceptanceContract("contract-v1", ("python check.py",))
            receipt = run_verifier("task", root, contract)
            self.assertTrue(
                receipt_is_current(
                    receipt,
                    task_id="task",
                    root=root,
                    contract=contract,
                    verifier_version="local-v1",
                )
            )
            (root / "check.py").write_text("print('changed')", encoding="utf-8")
            self.assertFalse(
                receipt_is_current(
                    receipt,
                    task_id="task",
                    root=root,
                    contract=contract,
                    verifier_version="local-v1",
                )
            )

    def test_failed_verifier_cannot_pass_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = AcceptanceContract(
                "contract-v1", ("python -c 'raise SystemExit(3)'",)
            )
            receipt = run_verifier("task", tmp, contract)
            self.assertEqual(receipt.outcome, "code_failed")
            self.assertFalse(
                receipt_is_current(
                    receipt,
                    task_id="task",
                    root=tmp,
                    contract=contract,
                    verifier_version="local-v1",
                )
            )

    def test_real_verifier_process_failures_have_distinct_outcomes(self):
        cases = (
            (
                AcceptanceContract(
                    "missing-command", ("horizon_harness_missing_verifier",)
                ),
                None,
                "infrastructure_failed",
            ),
            (
                AcceptanceContract("crashed-verifier", ("kill -TERM $$",)),
                None,
                "verifier_failed",
            ),
            (
                AcceptanceContract("budget", ("sleep 1",)),
                0.01,
                "budget_stopped",
            ),
        )
        for contract, timeout_seconds, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                if timeout_seconds is None:
                    receipt = run_verifier("task", tmp, contract)
                else:
                    receipt = run_verifier(
                        "task", tmp, contract, timeout_seconds=timeout_seconds
                    )
                self.assertEqual(receipt.outcome, expected)
