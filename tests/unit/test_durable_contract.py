import tempfile
import unittest
from pathlib import Path

from horizon_harness.tools.edit import edit_file
from horizon_harness.tools.read import read_file
from horizon_harness.workspace import TaskSession
from horizon_harness.llm.env import FAKE_PROVIDER


class DurableContractTests(unittest.TestCase):
    def test_default_workspace_session_does_not_expose_bash(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = TaskSession(
                FAKE_PROVIDER,
                code_dir=Path(tmp) / "code",
                store_dir=Path(tmp) / "store",
            )
            self.assertEqual(session.spec.capabilities, ("read", "edit"))

    def test_durable_session_rejects_bash_capability(self):
        from horizon_harness.spec import AgentSpec

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                TaskSession(
                    FAKE_PROVIDER,
                    code_dir=Path(tmp) / "code",
                    store_dir=Path(tmp) / "store",
                    spec=AgentSpec(name="x", capabilities=("bash",), workflow="single"),
                )

    def test_tools_reject_paths_outside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "code"
            root.mkdir()
            self.assertTrue(read_file("../secret", root=root).is_error)
            self.assertTrue(edit_file("../secret", "", "x", root=root).is_error)
