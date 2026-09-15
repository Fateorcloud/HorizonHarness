import tempfile
import unittest
from pathlib import Path

from horizon_harness.workspace.snapshot import (
    capture_tree,
    compare_tree,
    write_snapshot,
    materialize_snapshot,
)
from horizon_harness.workspace import TaskSession
from horizon_harness.llm.env import FAKE_PROVIDER


class WorkspaceSnapshotTests(unittest.TestCase):
    def test_task_checkpoint_publishes_snapshot_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            (root / "a.txt").write_text("x")
            store = Path(tmp) / "store"
            result = TaskSession(FAKE_PROVIDER, code_dir=root, store_dir=store).start(
                "read", pause_after_turns=1
            )
            checkpoint = result.record.checkpoint
            self.assertTrue(checkpoint.snapshot_path)
            self.assertTrue(
                (store / result.task_id / checkpoint.snapshot_path).is_file()
            )

    def test_capture_is_stable_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            (root / "a.txt").write_text("one", encoding="utf-8")
            snapshot = capture_tree(root)
            self.assertEqual(compare_tree(root, snapshot), ())
            (root / "a.txt").write_text("two", encoding="utf-8")
            (root / "b.txt").write_text("new", encoding="utf-8")
            self.assertEqual(
                compare_tree(root, snapshot), ("changed:a.txt", "added:b.txt")
            )

    def test_snapshot_export_is_reopenable_json(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            (root / "a.txt").write_text("one", encoding="utf-8")
            target = Path(tmp) / "snapshot.json"
            snapshot = capture_tree(root)
            write_snapshot(target, snapshot)
            self.assertEqual(
                json.loads(target.read_text())["tree_sha256"], snapshot.tree_sha256
            )

    def test_content_addressed_snapshot_materializes_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            (root / "a.txt").write_text("one")
            objects = Path(tmp) / "objects"
            snapshot = capture_tree(root, object_dir=objects)
            target = Path(tmp) / "materialized"
            materialize_snapshot(snapshot, object_dir=objects, target=target)
            self.assertEqual(capture_tree(target).tree_sha256, snapshot.tree_sha256)
            (objects / snapshot.entries[0].object_name).write_text("bad")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                materialize_snapshot(
                    snapshot, object_dir=objects, target=Path(tmp) / "bad-target"
                )
