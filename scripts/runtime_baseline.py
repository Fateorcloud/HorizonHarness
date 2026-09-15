"""Capture tracked changes and untracked source without copying credentials."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads((args.verify / "manifest.json").read_text())
        with zipfile.ZipFile(args.verify / "source.zip") as archive:
            for name, expected in manifest["sha256"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"baseline content mismatch: {name}")
        print(f"Verified {len(manifest['sha256'])} files; HEAD {manifest['head']}")
        return
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=root)

    destination = Path(tempfile.mkdtemp(prefix="horizon-runtime-baseline-"))
    tracked = git("ls-files", "-z").decode().split("\0")
    untracked = (
        git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    )
    selected = sorted(
        {
            name
            for name in tracked + untracked
            if name
            and (
                name.startswith(("src/", "tests/", "runs/", "scripts/", "docs/"))
                or name in {"README.md", "AGENTS.md", "pyproject.toml", "uv.lock"}
            )
        }
    )
    hashes = {}
    with zipfile.ZipFile(
        destination / "source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in selected:
            path = root / name
            if path.is_symlink() or not path.is_file():
                continue
            data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    (destination / "tracked.diff").write_bytes(git("diff", "--binary", "HEAD"))
    (destination / "status.txt").write_bytes(git("status", "--short"))
    manifest = {"head": git("rev-parse", "HEAD").decode().strip(), "sha256": hashes}
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
