"""Immutable, bounded snapshots for the controlled local code workspace."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FileEntry:
    path: str
    sha256: str
    mode: int
    size: int
    object_name: str = ""


@dataclass(frozen=True)
class FileTreeSnapshot:
    root: str
    entries: tuple[FileEntry, ...]
    tree_sha256: str

    def as_dict(self) -> dict:
        return {
            "root": self.root,
            "entries": [entry.__dict__ for entry in self.entries],
            "tree_sha256": self.tree_sha256,
        }


def capture_tree(
    root: str | Path,
    *,
    exclude: Iterable[str] = (),
    object_dir: str | Path | None = None,
) -> FileTreeSnapshot:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"snapshot root must be a directory: {root}")
    excluded = {Path(item).as_posix() for item in exclude}
    entries: list[FileEntry] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if relative in excluded:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        object_name = digest
        if object_dir is not None:
            objects = Path(object_dir).expanduser().resolve()
            objects.mkdir(parents=True, exist_ok=True)
            obj = objects / digest
            if not obj.exists():
                obj.write_bytes(path.read_bytes())
        entries.append(
            FileEntry(
                path=relative,
                sha256=digest,
                mode=os.stat(path).st_mode & 0o777,
                size=path.stat().st_size,
                object_name=object_name,
            )
        )
    encoded = "\n".join(
        f"{entry.path}\0{entry.sha256}\0{entry.mode:o}\0{entry.size}"
        for entry in entries
    ).encode()
    return FileTreeSnapshot(
        root=str(base),
        entries=tuple(entries),
        tree_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def compare_tree(root: str | Path, snapshot: FileTreeSnapshot) -> tuple[str, ...]:
    """Return deterministic drift descriptions without mutating the tree."""
    current = capture_tree(root)
    expected = {entry.path: entry for entry in snapshot.entries}
    actual = {entry.path: entry for entry in current.entries}
    changes: list[str] = []
    for path in sorted(set(expected) | set(actual)):
        if path not in expected:
            changes.append(f"added:{path}")
        elif path not in actual:
            changes.append(f"removed:{path}")
        elif expected[path] != actual[path]:
            changes.append(f"changed:{path}")
    return tuple(changes)


def write_snapshot(path: str | Path, snapshot: FileTreeSnapshot) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_text(
        json.dumps(snapshot.as_dict(), sort_keys=True, indent=2), encoding="utf-8"
    )
    os.replace(part, target)


def materialize_snapshot(
    snapshot: FileTreeSnapshot, *, object_dir: str | Path, target: str | Path
) -> None:
    """Materialize a content-addressed snapshot into an empty directory."""
    destination = Path(target).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("snapshot target must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    objects = Path(object_dir).expanduser().resolve()
    for entry in snapshot.entries:
        source = objects / (entry.object_name or entry.sha256)
        if (
            not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest() != entry.sha256
        ):
            raise ValueError(f"snapshot object missing or corrupt: {entry.path}")
        output = destination / entry.path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())
        os.chmod(output, entry.mode)
    if capture_tree(destination).tree_sha256 != snapshot.tree_sha256:
        raise ValueError("materialized snapshot tree hash mismatch")


def load_snapshot(path: str | Path) -> FileTreeSnapshot:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = tuple(FileEntry(**row) for row in value.get("entries", ()))
    snapshot = FileTreeSnapshot(
        root=str(value.get("root", "")),
        entries=entries,
        tree_sha256=str(value.get("tree_sha256", "")),
    )
    if not snapshot.tree_sha256:
        raise ValueError("snapshot manifest missing tree hash")
    return snapshot
