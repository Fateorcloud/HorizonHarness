"""Fixture inventory preflight for runtime regression experiments."""

from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
from pathlib import Path

from horizon_harness.workspace.snapshot import capture_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(
            "output directory already exists; refusing to overwrite results"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("manifest must be an object")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("manifest must contain a non-empty cases list")
    results = []
    seen: set[str] = set()
    validated: list[tuple[str, Path, str]] = []
    for case in cases:
        if not isinstance(case, dict) or "id" not in case or "root" not in case:
            raise SystemExit("each manifest case requires id and root")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise SystemExit(f"duplicate or empty case id: {case_id!r}")
        seen.add(case_id)
        if not isinstance(case["root"], str) or not case["root"].strip():
            raise SystemExit(f"case requires a fixture root path: {case_id}")
        root = Path(case["root"]).expanduser()
        if not root.is_absolute():
            root = args.manifest.resolve().parent / root
        root = root.resolve()
        if not root.is_dir():
            raise SystemExit(f"fixture root is not a directory: {root}")
        if case.get("debug_fork") or case.get("ancestry"):
            raise SystemExit(
                f"debug fork ancestry is not valid for formal regression: {case_id}"
            )
        command = case.get("verifier")
        if not isinstance(command, str) or not command.strip():
            raise SystemExit(f"case requires a verifier command: {case_id}")
        output_root = args.output_dir.resolve()
        if output_root.is_relative_to(root) or root.is_relative_to(output_root):
            raise SystemExit(
                "output directory must be separate from every fixture root"
            )
        validated.append((case_id, root, command))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for case_id, root, command in validated:
        completed = subprocess.run(
            command, cwd=root, shell=True, text=True, capture_output=True, check=False
        )
        output = completed.stdout + "\n" + completed.stderr
        results.append(
            {
                "id": case_id,
                "execution_contract": "deterministic_fixture",
                "evidence_kind": "deterministic_verifier",
                "root": str(root),
                "tree_sha256": capture_tree(root).tree_sha256,
                "verifier": command,
                "exit_code": completed.returncode,
                "output_digest": hashlib.sha256(output.encode()).hexdigest(),
                "status": "passed" if completed.returncode == 0 else "failed",
            }
        )
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
