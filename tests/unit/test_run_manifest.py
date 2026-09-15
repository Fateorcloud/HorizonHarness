"""Unified launch-manifest integration tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from runs.lib.run_manifest import start_run_manifest


class RunManifestTests(unittest.TestCase):
    def test_start_freezes_safe_inputs_before_run_without_serializing_secrets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dotenv = root / ".env.glm"
            dotenv.write_text(
                "\n".join(
                    (
                        "OPENAI_MODEL=z-ai/glm-5.2:free",
                        "OPENAI_AUTH_TOKEN=super-secret-token",
                        "OPENAI_BASE_URL=https://openrouter.ai/api/v1",
                        "HORIZON_API_KIND=openai-chat",
                        "HORIZON_CAPABILITIES=bash,read,task",
                        "HORIZON_WORKFLOW=single",
                    )
                ),
                encoding="utf-8",
            )
            args = Namespace(
                provider="openai",
                dotenv=str(dotenv),
                job_name="glm-pilot",
                jobs_dir=str(root / "jobs"),
                dataset="terminal-bench/terminal-bench-2-1",
                include_task_names=["task-a", "task-b"],
                max_turns=150,
                capabilities="bash,read,task",
                workflow="single",
                api_kind=None,
                agent_env=["OPENAI_AUTH_TOKEN=another-secret"],
            )
            handle = start_run_manifest(
                "harbor",
                args,
                root=root,
                environ={},
                started_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
                git_state=("abc123", True),
            )

            self.assertTrue(handle.sidecar_path.is_file())
            payload = json.loads(handle.sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "horizon-harness.run-manifest.v1")
            self.assertEqual(payload["run_id"], "glm-pilot")
            self.assertEqual(payload["source"]["git_commit"], "abc123")
            self.assertTrue(payload["source"]["git_dirty"])
            self.assertEqual(payload["model"]["requested_id"], "z-ai/glm-5.2:free")
            self.assertEqual(payload["model"]["endpoint_host"], "openrouter.ai")
            self.assertEqual(
                payload["task_selection"]["include_task_names"],
                ["task-a", "task-b"],
            )
            serialized = json.dumps(payload)
            self.assertNotIn("super-secret-token", serialized)
            self.assertNotIn("another-secret", serialized)
            self.assertNotIn("agent_env", serialized)

    def test_attach_copies_manifest_into_existing_adapter_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                provider="fake",
                run_id="fake-run",
                run_root=str(root / "runs"),
                instance_id="case-1",
                max_turns=3,
                capabilities="bash,read,task",
                workflow="single",
                api_kind="fake",
                dotenv=None,
            )
            handle = start_run_manifest(
                "swebench",
                args,
                root=root,
                environ={},
                started_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
                git_state=("abc123", False),
            )
            run_dir = root / "runs/fake-run"
            run_dir.mkdir(parents=True)

            outcome = handle.attach(
                {"bench": "swebench", "status_code": 0, "run_dir": str(run_dir)}
            )

            attached = run_dir / "run-manifest.json"
            self.assertEqual(outcome["run_manifest_path"], str(attached))
            self.assertEqual(
                json.loads(attached.read_text(encoding="utf-8")), handle.payload
            )

    def test_terminus_baseline_manifest_records_reference_agent_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                provider="openai",
                harbor_agent="terminus-2",
                dotenv=None,
                job_name="baseline-shard-key1",
                jobs_dir=str(root / "jobs"),
                dataset="terminal-bench/terminal-bench-2-1",
                include_task_names=["terminal-bench/task-a"],
                max_turns=150,
                max_output_tokens=4096,
                n_concurrent=2,
                model=["openai/deepseek-ai/deepseek-v4-pro-0813"],
                api_kind="openai-chat",
            )

            handle = start_run_manifest(
                "harbor",
                args,
                root=root,
                environ={
                    "OPENAI_MODEL": "deepseek-ai/deepseek-v4-pro-0813",
                    "OPENAI_BASE_URL": "https://integrate.api.nvidia.com/v1",
                },
                started_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
                git_state=("abc123", False),
            )

        self.assertEqual(handle.payload["agent"]["name"], "terminus-2")
        self.assertEqual(handle.payload["agent"]["capabilities"], ["terminal"])
        self.assertEqual(handle.payload["agent"]["workflow"], "terminus-2")
        self.assertFalse(handle.payload["agent"]["context_summarization"])


if __name__ == "__main__":
    unittest.main()
