"""Acceptance evidence must describe frozen tests and an unchanged code tree."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from horizon_harness.llm.env import FAKE_PROVIDER
from horizon_harness.llm.types import LLMResponse
from horizon_harness.messages import TextBlock
from horizon_harness.workspace import TaskSession
from horizon_harness.workspace.models import (
    VerifierResult,
    WorkspaceRecord,
    record_from_dict,
)
from horizon_harness.workspace.replay import fork_from_checkpoint
from horizon_harness.workspace.snapshot import (
    capture_tree,
    compare_tree,
)
from horizon_harness.workspace.store import UnsafeCheckpointError
from horizon_harness.workspace.verification import (
    AcceptanceContract,
    VerificationReceipt,
    receipt_is_current,
    run_verifier,
)
from runs.task_workspace import main


CACHE_DIRS = (".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")


class AcceptanceIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.code = self.root / "code"
        self.code.mkdir()
        self.acceptance = self.root / "acceptance"
        self.acceptance.mkdir()
        (self.acceptance / "check.py").write_text("pass\n", encoding="utf-8")
        self.store = self.root / "store"
        self.session = self.new_session()

    def new_session(self, **kwargs) -> TaskSession:
        return TaskSession(
            FAKE_PROVIDER,
            code_dir=kwargs.pop("code_dir", self.code),
            store_dir=kwargs.pop("store_dir", self.store),
            complete_fn=lambda request: LLMResponse(
                content=(TextBlock("Ready for verification."),),
                stop_reason="end_turn",
                model="fake-model",
            ),
            **kwargs,
        )

    def start(self, commands=None, **kwargs):
        contract = AcceptanceContract(
            "integrity",
            commands or ('python "$HORIZON_ACCEPTANCE_DIR/check.py"',),
            acceptance_dir=kwargs.pop("acceptance_dir", self.acceptance),
            **kwargs,
        )
        task = self.session.start(
            "Check the implementation",
            acceptance_contract=contract,
            pause_after_turns=1,
        )
        return task.task_id, contract

    def current(self, receipt, contract, **kwargs) -> bool:
        return receipt_is_current(
            receipt,
            task_id=receipt.task_id,
            root=self.code,
            contract=contract,
            verifier_version="local-v1",
            task_revision=receipt.task_revision,
            **kwargs,
        )

    def frozen_contract(self, task_id, contract):
        return replace(
            contract,
            acceptance_tree_sha256=self.session.inspect(task_id).acceptance_tree_sha256,
        )

    def test_frozen_tests_resist_original_and_workspace_replacements(self):
        (self.code / "answer.txt").write_text("wrong")
        (self.acceptance / "check.py").write_text(
            "from pathlib import Path\nassert Path('answer.txt').read_text() == 'right'\n"
        )
        task_id, contract = self.start()
        frozen = self.session.inspect(task_id)
        (self.acceptance / "check.py").write_text("pass\n")
        (self.code / "check.py").write_text("pass\n")

        self.assertEqual(self.session.verify(task_id, contract).outcome, "code_failed")
        (self.code / "answer.txt").write_text("right")
        receipt = self.session.verify(task_id, contract)
        self.assertEqual(receipt.outcome, "passed")
        self.assertEqual(receipt.acceptance_tree_sha256, frozen.acceptance_tree_sha256)
        self.assertEqual(
            self.session.complete_if_verified(task_id, contract).status, "completed"
        )
        self.assertEqual(
            self.session.inspect(task_id).acceptance_snapshot,
            frozen.acceptance_snapshot,
        )

    def test_temp_directory_is_external_unique_and_removed_on_both_outcomes(self):
        observed = self.root / "observed.json"
        (self.acceptance / "check.py").write_text(
            "import json, os\nfrom pathlib import Path\n"
            f"Path({str(observed)!r}).write_text(json.dumps(dict(path=os.environ['HORIZON_ACCEPTANCE_DIR'], cwd=str(Path.cwd()))))\n"
            "raise SystemExit(int(Path('exit.txt').read_text()))\n"
        )
        (self.code / "exit.txt").write_text("0")
        task_id, contract = self.start()
        paths = []
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code):
                (self.code / "exit.txt").write_text(str(exit_code))
                before = capture_tree(self.code)
                receipt = self.session.verify(task_id, contract)
                self.assertEqual(
                    receipt.outcome, "passed" if exit_code == 0 else "code_failed"
                )
                observation = json.loads(observed.read_text())
                temporary = Path(observation["path"])
                paths.append(temporary)
                self.assertFalse(temporary.is_relative_to(self.code))
                self.assertFalse(temporary.is_relative_to(self.store))
                self.assertFalse(temporary.exists())
                self.assertEqual(observation["cwd"], str(self.code))
                self.assertEqual(capture_tree(self.code), before)
        self.assertNotEqual(*paths)

    def test_deleted_original_acceptance_directory_does_not_affect_verification(self):
        task_id, contract = self.start()
        (self.acceptance / "check.py").unlink()
        self.acceptance.rmdir()
        self.assertEqual(self.session.verify(task_id, contract).outcome, "passed")
        self.assertEqual(
            self.session.complete_if_verified(task_id, contract).status, "completed"
        )

    def test_corrupt_and_missing_objects_block_command_and_completion(self):
        marker = self.root / "command-ran"
        for damage in ("corrupt", "missing"):
            with self.subTest(damage=damage):
                task_id, contract = self.start((f"touch '{marker}'",))
                record = self.session.inspect(task_id)
                entry = record.acceptance_snapshot.entries[0]
                obj = (
                    self.session.store.task_dir(task_id)
                    / "snapshot-objects"
                    / entry.object_name
                )
                if damage == "corrupt":
                    obj.write_bytes(b"corrupt")
                else:
                    obj.unlink()
                receipt = self.session.verify(task_id, contract)
                self.assertEqual(receipt.outcome, "acceptance_corrupted")
                self.assertEqual(
                    receipt.acceptance_tree_sha256, record.acceptance_tree_sha256
                )
                self.assertFalse(marker.exists())
                with self.assertRaisesRegex(UnsafeCheckpointError, "stale or failed"):
                    self.session.complete_if_verified(task_id, contract)

    def test_materialization_hash_mismatch_never_runs_command(self):
        marker = self.root / "command-ran"
        task_id, contract = self.start((f"touch '{marker}'",))
        record = self.session.inspect(task_id)
        snapshot = record.acceptance_snapshot
        broken = replace(snapshot, entries=(replace(snapshot.entries[0], mode=0o700),))
        receipt = run_verifier(
            task_id,
            self.code,
            self.frozen_contract(task_id, contract),
            acceptance_snapshot=broken,
            object_dir=self.session.store.task_dir(task_id) / "snapshot-objects",
            store_dir=self.store,
        )
        self.assertEqual(receipt.outcome, "acceptance_corrupted")
        self.assertFalse(marker.exists())

    def test_partial_materialization_failure_cleans_up_without_workspace_files(self):
        task_id, contract = self.start()
        before = capture_tree(self.code)
        targets = []

        def fail_materialization(snapshot, *, object_dir, target):
            targets.append(Path(target))
            (Path(target) / "partial").write_text("partial")
            raise OSError("injected materialization failure")

        with (
            patch(
                "horizon_harness.workspace.verification.materialize_snapshot",
                side_effect=fail_materialization,
            ),
            patch("horizon_harness.workspace.verification.subprocess.run") as command,
        ):
            receipt = self.session.verify(task_id, contract)
        self.assertEqual(receipt.outcome, "acceptance_corrupted")
        command.assert_not_called()
        self.assertTrue(targets)
        self.assertTrue(all(not target.exists() for target in targets))
        self.assertEqual(capture_tree(self.code), before)

    def test_directory_and_hash_mismatches_reject_verify_and_complete(self):
        task_id, contract = self.start()
        self.session.verify(task_id, contract)
        alternatives = (
            replace(contract, acceptance_dir=self.root / "other"),
            replace(contract, acceptance_dir=None),
            replace(contract, acceptance_tree_sha256="0" * 64),
            replace(contract, acceptance_tree_sha256=""),
            replace(contract, verification_exclude_dirs=("build",)),
        )
        for supplied in alternatives:
            for action in (self.session.verify, self.session.complete_if_verified):
                with (
                    self.subTest(contract=supplied, action=action.__name__),
                    self.assertRaisesRegex(UnsafeCheckpointError, "frozen"),
                ):
                    action(task_id, supplied)
        self.assertEqual(
            self.session.complete_if_verified(
                task_id, self.frozen_contract(task_id, contract)
            ).status,
            "completed",
        )

    def test_overlapping_acceptance_directories_are_rejected_at_creation(self):
        nested = self.code / "tests"
        nested.mkdir()
        (nested / "check.py").write_text("pass")
        for directory in (self.code, nested, self.root):
            with (
                self.subTest(directory=directory),
                self.assertRaisesRegex(ValueError, "acceptance_dir.*outside|overlap"),
            ):
                self.start(acceptance_dir=directory)

    def test_missing_empty_symlink_and_special_acceptance_entries_are_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        link = self.root / "linked"
        link.symlink_to(self.acceptance, target_is_directory=True)
        for directory in (self.root / "missing", empty, link):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                self.start(acceptance_dir=directory)
        for target in (
            self.acceptance / "file-link",
            self.acceptance / "dir-link",
            self.acceptance / "dangling",
        ):
            target.symlink_to(
                self.code if target.name == "dir-link" else self.root / "missing"
            )
            with (
                self.subTest(target=target),
                self.assertRaisesRegex(ValueError, "symlink"),
            ):
                self.start()
            target.unlink()
        fifo = self.acceptance / "pipe"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "regular"):
            self.start()

    def test_declared_hash_is_checked_when_freezing(self):
        with self.assertRaisesRegex(ValueError, "hash"):
            self.start(acceptance_tree_sha256="0" * 64)
        expected = capture_tree(self.acceptance).tree_sha256
        task_id, contract = self.start(acceptance_tree_sha256=expected)
        self.assertEqual(self.session.inspect(task_id).acceptance_tree_sha256, expected)
        self.assertEqual(self.session.verify(task_id, contract).outcome, "passed")

    def test_frozen_manifest_hash_directory_and_rules_cannot_be_saved_over(self):
        task_id, _ = self.start()
        record = self.session.inspect(task_id)
        mutations = (
            dict(acceptance_dir=str(self.root / "other")),
            dict(acceptance_tree_sha256="0" * 64),
            dict(acceptance_snapshot=replace(record.acceptance_snapshot, entries=())),
            dict(verification_exclude_dirs=("build",)),
        )
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(ValueError, "frozen"),
            ):
                self.session.store.save(replace(record, **mutation))
        self.assertEqual(self.session.inspect(task_id), record)

    def test_receipt_acceptance_hash_is_persisted_and_checked(self):
        task_id, contract = self.start()
        contract = self.frozen_contract(task_id, contract)
        receipt = self.session.verify(task_id, contract)
        self.assertTrue(self.current(receipt, contract))
        self.assertFalse(
            self.current(replace(receipt, acceptance_tree_sha256="0" * 64), contract)
        )
        self.assertFalse(
            self.current(receipt, replace(contract, acceptance_tree_sha256="0" * 64))
        )
        reopened = self.new_session().inspect(task_id)
        self.assertEqual(
            reopened.verifier_results[-1].acceptance_tree_sha256,
            receipt.acceptance_tree_sha256,
        )
        exported = json.loads(
            (self.session.store.task_dir(task_id) / "task.json").read_text()
        )
        self.assertEqual(
            exported["acceptance_snapshot"]["tree_sha256"],
            receipt.acceptance_tree_sha256,
        )
        self.assertEqual(record_from_dict(exported), reopened)

    @unittest.skipUnless(importlib.util.find_spec("pytest"), "requires pytest")
    def test_real_pytest_caches_keep_receipt_current_and_allow_completion(self):
        (self.code / "solution.py").write_text("answer = 42\n")
        (self.acceptance / "test_solution.py").write_text(
            "import importlib.util\nfrom pathlib import Path\n"
            "def test_answer():\n"
            "    spec = importlib.util.spec_from_file_location('solution', Path.cwd() / 'solution.py')\n"
            "    module = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(module)\n"
            "    assert module.answer == 42\n"
        )
        (self.acceptance / "pytest.ini").write_text(
            f"[pytest]\ncache_dir = {self.code}/.pytest_cache\n"
        )
        task_id, contract = self.start(('python -m pytest "$HORIZON_ACCEPTANCE_DIR"',))
        receipt = self.session.verify(task_id, contract)
        self.assertEqual(receipt.outcome, "passed")
        self.assertTrue((self.code / "__pycache__").is_dir())
        self.assertTrue((self.code / ".pytest_cache").is_dir())
        self.assertEqual(receipt.workspace_changes, ())
        self.assertTrue(self.current(receipt, self.frozen_contract(task_id, contract)))
        self.assertEqual(
            self.session.complete_if_verified(task_id, contract).status, "completed"
        )

    def test_default_exclusions_match_directory_components_only(self):
        for name in CACHE_DIRS:
            directory = self.code / "nested" / name
            directory.mkdir(parents=True)
            (directory / "cached").write_text("before")
        same_named_file = self.code / ".git"
        same_named_file.write_text("gitdir: /elsewhere")
        prefix_file = self.code / ".pytest_cache.txt"
        prefix_file.write_text("before")
        contract = AcceptanceContract("plain", ("true",))
        receipt = run_verifier("task", self.code, contract)
        self.assertEqual(set(receipt.verification_exclude_dirs), set(CACHE_DIRS))
        for name in CACHE_DIRS:
            (self.code / "nested" / name / "cached").write_text("after")
        self.assertTrue(self.current(receipt, contract))
        for file in (same_named_file, prefix_file):
            before = file.read_text()
            file.write_text("changed")
            self.assertFalse(self.current(receipt, contract))
            file.write_text(before)

    def test_custom_exclusions_are_frozen_persisted_and_bound_to_receipt(self):
        task_id, contract = self.start(
            ("mkdir -p nested/build; touch nested/build/cache",),
            verification_exclude_dirs=("build",),
        )
        receipt = self.session.verify(task_id, contract)
        self.assertEqual(receipt.outcome, "passed")
        self.assertIn("build", receipt.verification_exclude_dirs)
        self.assertEqual(
            self.session.inspect(task_id).verification_exclude_dirs,
            receipt.verification_exclude_dirs,
        )
        self.assertFalse(
            self.current(
                replace(receipt, verification_exclude_dirs=("different",)),
                self.frozen_contract(task_id, contract),
            )
        )
        self.assertEqual(
            self.session.complete_if_verified(task_id, contract).status, "completed"
        )

    def test_exclusion_rules_require_plain_directory_names(self):
        for name in ("", ".", "..", "a/b", "a\\b", "/tmp"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                AcceptanceContract(
                    "bad-rule", ("true",), verification_exclude_dirs=(name,)
                )

    def test_legacy_receipt_without_rules_hashes_cache_files_too(self):
        cache = self.code / "__pycache__"
        cache.mkdir()
        (cache / "before").write_text("cache")
        contract = AcceptanceContract("legacy", ("true",))
        value = asdict(run_verifier("legacy", self.code, contract))
        for key in (
            "acceptance_tree_sha256",
            "verification_exclude_dirs",
            "workspace_changes",
        ):
            value.pop(key)
        value["tree_sha256"] = capture_tree(self.code).tree_sha256
        legacy = VerificationReceipt(**value)
        self.assertTrue(self.current(legacy, contract))
        (cache / "after").write_text("new cache")
        self.assertFalse(self.current(legacy, contract))

    def test_legacy_task_and_receipt_still_complete_without_acceptance_directory(self):
        contract = AcceptanceContract("legacy", ("true",))
        record = WorkspaceRecord(
            1,
            "legacy",
            0,
            "old task",
            acceptance_contract_id=contract.contract_id,
            acceptance_contract_version=1,
            acceptance_commands=contract.required_commands,
        )
        value = record.as_dict()
        for key in (
            "acceptance_dir",
            "acceptance_tree_sha256",
            "acceptance_snapshot",
            "verification_exclude_dirs",
        ):
            value.pop(key)
        record = record_from_dict(value)
        self.session.store.save(record)
        receipt = run_verifier("legacy", self.code, contract, task_revision=1)
        old_result = VerifierResult(
            command=receipt.command,
            passed=True,
            exit_code=0,
            evidence_ref=receipt.output_digest,
            tree_sha256=capture_tree(self.code).tree_sha256,
            contract_id=contract.contract_id,
            contract_version=1,
            verifier_version="local-v1",
            task_revision=1,
            outcome="passed",
        )
        self.session.store.save(
            replace(record, revision=1, verifier_results=(old_result,))
        )
        self.assertEqual(
            self.session.complete_if_verified("legacy", contract).status, "completed"
        )

    def test_workspace_writes_report_added_modified_removed_without_undo(self):
        (self.code / "changed.txt").write_text("before")
        (self.code / "removed.txt").write_text("remove me")
        before = capture_tree(self.code).tree_sha256
        task_id, contract = self.start(
            ("printf new > added.txt; printf after > changed.txt; rm removed.txt",)
        )
        receipt = self.session.verify(task_id, contract)
        self.assertEqual(receipt.exit_code, 0)
        self.assertEqual(receipt.outcome, "verifier_modified_workspace")
        self.assertEqual(receipt.tree_sha256, before)
        self.assertEqual(
            receipt.workspace_changes,
            ("added:added.txt", "changed:changed.txt", "removed:removed.txt"),
        )
        self.assertEqual((self.code / "added.txt").read_text(), "new")
        self.assertEqual((self.code / "changed.txt").read_text(), "after")
        self.assertFalse((self.code / "removed.txt").exists())
        self.assertEqual(
            self.session.inspect(task_id).verifier_results[-1].workspace_changes,
            receipt.workspace_changes,
        )
        self.assertFalse(self.session.inspect(task_id).verifier_results[-1].passed)
        with self.assertRaisesRegex(UnsafeCheckpointError, "stale or failed"):
            self.session.complete_if_verified(task_id, contract)

    def test_workspace_mutation_priority_preserves_process_failures(self):
        cases = (
            (0, "verifier_modified_workspace"),
            (1, "verifier_modified_workspace"),
            (126, "infrastructure_failed"),
            (127, "infrastructure_failed"),
            (129, "verifier_failed"),
        )
        for exit_code, outcome in cases:
            with self.subTest(exit_code=exit_code):
                target = f"written-{exit_code}"
                receipt = run_verifier(
                    "task",
                    self.code,
                    AcceptanceContract(
                        "priority", (f"touch {target}; exit {exit_code}",)
                    ),
                )
                self.assertEqual(receipt.outcome, outcome)
                self.assertEqual(receipt.workspace_changes, (f"added:{target}",))
        receipt = run_verifier(
            "task",
            self.code,
            AcceptanceContract("timeout", ("touch timed-out; sleep 0.1",)),
            timeout_seconds=0.02,
        )
        self.assertEqual(receipt.outcome, "budget_stopped")
        self.assertEqual(receipt.workspace_changes, ("added:timed-out",))

    def test_subprocess_exception_still_captures_workspace_changes(self):
        def fail(*args, **kwargs):
            (self.code / "side-effect").write_text("left in place")
            raise OSError("injected spawn failure")

        with patch(
            "horizon_harness.workspace.verification.subprocess.run", side_effect=fail
        ):
            receipt = run_verifier(
                "task", self.code, AcceptanceContract("failure", ("true",))
            )
        self.assertEqual(receipt.outcome, "infrastructure_failed")
        self.assertEqual(receipt.workspace_changes, ("added:side-effect",))

    def test_acceptance_environment_does_not_leak_into_parent(self):
        task_id, contract = self.start()
        with patch.dict(os.environ, {"HORIZON_ACCEPTANCE_DIR": "parent-value"}):
            self.assertEqual(self.session.verify(task_id, contract).outcome, "passed")
            self.assertEqual(os.environ["HORIZON_ACCEPTANCE_DIR"], "parent-value")

    def test_temp_parent_inside_code_or_store_is_not_used(self):
        observed = self.root / "temp-path"
        (self.acceptance / "check.py").write_text(
            "import os\nfrom pathlib import Path\n"
            f"Path({str(observed)!r}).write_text(os.environ['HORIZON_ACCEPTANCE_DIR'])\n"
        )
        task_id, contract = self.start()
        for parent in (self.code, self.store):
            with (
                self.subTest(parent=parent),
                patch(
                    "horizon_harness.workspace.verification.tempfile.gettempdir",
                    return_value=str(parent),
                ),
            ):
                self.assertEqual(
                    self.session.verify(task_id, contract).outcome, "passed"
                )
                temporary = Path(observed.read_text())
                self.assertFalse(temporary.is_relative_to(self.code))
                self.assertFalse(temporary.is_relative_to(self.store))
                self.assertFalse(temporary.exists())

    def test_workspace_mutation_receipt_stays_failed_after_manual_restore(self):
        task_id, contract = self.start(("touch written-by-verifier",))
        receipt = self.session.verify(task_id, contract)
        self.assertEqual(receipt.outcome, "verifier_modified_workspace")
        (self.code / "written-by-verifier").unlink()
        self.assertEqual(capture_tree(self.code).tree_sha256, receipt.tree_sha256)
        with self.assertRaisesRegex(UnsafeCheckpointError, "stale or failed"):
            self.session.complete_if_verified(task_id, contract)

    def test_checkpoint_and_restore_retain_unfiltered_cache_and_git_hashes(self):
        for name in CACHE_DIRS:
            directory = self.code / name
            directory.mkdir()
            (directory / "state").write_text(name)
            (directory / "state").chmod(0o640)
        before = capture_tree(self.code)
        self.assertEqual(
            before.tree_sha256,
            "19c916b5c0d9bd0d98fb29af08ce7ae2ce8c2554780c698c05ed079cb5d0af62",
        )
        task_id, contract = self.start(("true",))
        record = self.session.inspect(task_id)
        self.assertEqual(record.checkpoint.workspace_tree_sha256, before.tree_sha256)
        receipt = self.session.verify(task_id, contract)
        self.assertNotEqual(receipt.tree_sha256, before.tree_sha256)
        target = self.root / "restored"
        source_hash, restored_hash = fork_from_checkpoint(
            self.session.store.task_dir(task_id),
            record.checkpoint.snapshot_path,
            target,
        )
        self.assertEqual(
            (source_hash, restored_hash), (before.tree_sha256, before.tree_sha256)
        )
        self.assertEqual(capture_tree(target).tree_sha256, before.tree_sha256)
        for name in CACHE_DIRS:
            self.assertEqual((target / name / "state").read_text(), name)
        (self.code / "__pycache__" / "state").write_text("drift")
        self.assertEqual(
            compare_tree(self.code, before), ("changed:__pycache__/state",)
        )

    def test_fork_inherits_frozen_acceptance_objects_and_rules(self):
        task_id, contract = self.start(verification_exclude_dirs=("build",))
        (self.acceptance / "check.py").write_text("raise SystemExit(1)")
        target = self.root / "fork-code"
        child_store = self.root / "fork-store"
        child = self.session.fork_task(
            task_id,
            child_task_id="child",
            child_code_dir=target,
            child_store_dir=child_store,
        )
        parent = self.session.inspect(task_id)
        self.assertEqual(child.acceptance_snapshot, parent.acceptance_snapshot)
        self.assertEqual(child.acceptance_tree_sha256, parent.acceptance_tree_sha256)
        self.assertEqual(
            child.verification_exclude_dirs, parent.verification_exclude_dirs
        )
        child_session = self.new_session(code_dir=target, store_dir=child_store)
        self.assertEqual(child_session.verify("child", contract).outcome, "passed")
        self.assertEqual(
            child_session.complete_if_verified("child", contract).status, "completed"
        )

    def test_cli_reads_frozen_directory_hash_and_rules_for_verify_and_complete(self):
        common = [
            "--fake",
            "--code-dir",
            str(self.code),
            "--store-dir",
            str(self.store),
        ]
        check = 'python "$HORIZON_ACCEPTANCE_DIR/check.py"'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        *common,
                        "start",
                        "check",
                        "--task-id",
                        "cli",
                        "--check",
                        check,
                        "--acceptance-dir",
                        str(self.acceptance),
                        "--verification-exclude-dir",
                        "build",
                        "--pause-after-turns",
                        "1",
                    ]
                ),
                0,
            )
            (self.acceptance / "check.py").write_text("raise SystemExit(1)")
            self.assertEqual(main([*common, "verify", "cli", "--check", check]), 0)
            self.assertEqual(main([*common, "complete", "cli", "--check", check]), 0)
        record = self.session.inspect("cli")
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.acceptance_dir, str(self.acceptance))
        self.assertIn("build", record.verification_exclude_dirs)


if __name__ == "__main__":
    unittest.main()
