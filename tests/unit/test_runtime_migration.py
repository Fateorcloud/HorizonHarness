import tempfile
import unittest
import sqlite3
from dataclasses import replace
from pathlib import Path

from horizon_harness.workspace.legacy import (
    legacy_requires_import,
    read_legacy_record,
    import_legacy_record,
)
from horizon_harness.workspace.store import WorkspaceStore
from horizon_harness.workspace import TaskSession
from horizon_harness.llm.env import FAKE_PROVIDER
from horizon_harness.tools import tool_arguments_hash
from horizon_harness.tools.edit import edit_fingerprint


class RuntimeMigrationTests(unittest.TestCase):
    def test_mutable_batch_progress_without_result_facts_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.sqlite3"
            with sqlite3.connect(path) as db:
                db.execute(
                    "CREATE TABLE tool_batches (batch_id TEXT PRIMARY KEY, "
                    "turn_no INTEGER NOT NULL, calls_json TEXT NOT NULL, "
                    "next_tool_index INTEGER NOT NULL, "
                    "bundle_committed INTEGER NOT NULL DEFAULT 0)"
                )
                db.execute(
                    "CREATE TABLE tool_batch_results (batch_id TEXT NOT NULL, "
                    "call_index INTEGER NOT NULL, result_json TEXT NOT NULL, "
                    "PRIMARY KEY (batch_id, call_index))"
                )
                db.execute(
                    "INSERT INTO tool_batches VALUES "
                    '(\'legacy\', 1, \'{"calls": [{"id": "A"}]}\', 1, 0)'
                )
            from horizon_harness.workspace.journal import ExecutionJournal

            with self.assertRaisesRegex(ValueError, "LEGACY_MUTABLE_BATCH_PROGRESS"):
                ExecutionJournal(path)

    def test_unscoped_legacy_bundle_marker_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.sqlite3"
            with sqlite3.connect(path) as db:
                db.execute(
                    "CREATE TABLE tool_batches (batch_id TEXT PRIMARY KEY, "
                    "turn_no INTEGER NOT NULL, calls_json TEXT NOT NULL, "
                    "next_tool_index INTEGER NOT NULL, "
                    "bundle_committed INTEGER NOT NULL DEFAULT 0)"
                )
                db.execute(
                    "CREATE TABLE tool_batch_results (batch_id TEXT NOT NULL, "
                    "call_index INTEGER NOT NULL, result_json TEXT NOT NULL, "
                    "PRIMARY KEY (batch_id, call_index))"
                )
                db.execute(
                    "INSERT INTO tool_batches VALUES "
                    "('legacy', 1, '{\"calls\": []}', 0, 1)"
                )
            from horizon_harness.workspace.journal import ExecutionJournal

            with self.assertRaisesRegex(ValueError, "LEGACY_UNSCOPED_BUNDLE_COMMIT"):
                ExecutionJournal(path)

    def test_legacy_record_is_read_only_and_requires_explicit_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = WorkspaceStore(tmp).create("legacy", task_id="old")
            path = Path(tmp) / "old" / "task.json"
            loaded = read_legacy_record(path)
            self.assertEqual(loaded.task_id, record.task_id)
            self.assertTrue(legacy_requires_import(path))

    def test_import_creates_new_paused_task_without_old_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_store = WorkspaceStore(Path(tmp) / "old")
            old = old_store.create("legacy", task_id="old")
            imported = import_legacy_record(
                Path(tmp) / "old" / "old" / "task.json",
                store=WorkspaceStore(Path(tmp) / "new"),
                task_id="new",
            )
            self.assertNotEqual(imported.task_id, old.task_id)
            self.assertEqual(imported.status, "paused")
            self.assertEqual(imported.verifier_results, ())

    def test_public_resume_rejects_legacy_pending_batch_without_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            store = WorkspaceStore(Path(tmp) / "store")
            record = store.create("legacy pending", task_id="legacy-pending")
            store.journal(record.task_id).plan_batch(
                batch_id="old-batch",
                turn_no=1,
                calls=[{"id": "call-a", "name": "edit", "arguments": {}}],
            )
            resumable = replace(
                record,
                status="paused",
                checkpoint=replace(record.checkpoint, resumable=True),
            )
            store.save(resumable)
            model_calls = []

            def model(request):
                model_calls.append(request)
                raise AssertionError("legacy recovery must not call the model")

            with self.assertRaisesRegex(
                ValueError, "MISSING_DURABLE_ASSISTANT_MESSAGE"
            ):
                TaskSession(
                    FAKE_PROVIDER,
                    code_dir=root,
                    store_dir=Path(tmp) / "store",
                    complete_fn=model,
                ).resume("legacy-pending")
            self.assertEqual(model_calls, [])

    def test_legacy_validation_precedes_edit_reconciliation_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            target = root / "value.txt"
            target.write_text("before", encoding="utf-8")
            store = WorkspaceStore(Path(tmp) / "store")
            record = store.create("legacy pending", task_id="legacy-side-effect")
            journal = store.journal(record.task_id)
            arguments = {
                "path": "value.txt",
                "old_string": "before",
                "new_string": "after",
                "replace_all": False,
            }
            journal.plan_batch(
                batch_id="old-batch",
                turn_no=1,
                calls=[
                    {
                        "id": "call-a",
                        "execution_step_id": "old-step",
                        "name": "edit",
                        "arguments": arguments,
                    }
                ],
            )
            journal.plan_step(
                step_id="old-step",
                tool_name="edit",
                input_hash=tool_arguments_hash(arguments),
                idempotency_key="old-step:v1",
                effect_class="idempotent_effect",
                result_payload={
                    "path": "value.txt",
                    "before": edit_fingerprint("before"),
                    "after": edit_fingerprint("after"),
                    "old_string": "before",
                    "new_string": "after",
                    "replace_all": False,
                },
            )
            resumable = replace(
                record,
                status="paused",
                checkpoint=replace(record.checkpoint, resumable=True),
            )
            store.save(resumable)

            with self.assertRaisesRegex(
                ValueError, "MISSING_DURABLE_ASSISTANT_MESSAGE"
            ):
                TaskSession(
                    FAKE_PROVIDER,
                    code_dir=root,
                    store_dir=Path(tmp) / "store",
                    complete_fn=lambda request: (_ for _ in ()).throw(
                        AssertionError("legacy recovery must not call the model")
                    ),
                ).resume("legacy-side-effect")

            self.assertEqual(target.read_text(encoding="utf-8"), "before")
