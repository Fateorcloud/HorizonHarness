"""Prepare, smoke-test, and report the optional indexed-memory backend."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from horizon_harness.memory import (
    IndexedMemory,
    MemoryCandidate,
    SentenceTransformerEmbedder,
)


MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Validate keyword/hybrid retrieval; hybrid may download the optional model."
    )
    root.add_argument(
        "--db", type=Path, default=Path("evals/out/memory-validation/memory.sqlite3")
    )
    sub = root.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare",
        help=(
            "load explicit candidates for storage/retrieval fixtures only; "
            "real task extraction is performed by runs.task_workspace"
        ),
    )
    prepare.add_argument("--manifest", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=("keyword", "hybrid"), required=True)
    run.add_argument("--query", required=True)
    run.add_argument("--scope", required=True)
    run.add_argument("--revision", help="exact Hugging Face commit required for hybrid")
    run.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("inputs", type=Path, nargs="+")
    report.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        memory = IndexedMemory(args.db)
        source = data["source"]
        memory.register_sources(source["task_id"], source["run_id"], source["refs"])
        candidates = tuple(MemoryCandidate(**row) for row in data["candidates"])
        ids = memory.finish_task(
            task_id=source["task_id"],
            source_revision=source["revision"],
            outcome=source["outcome"],
            candidates=candidates,
        )
        print(
            json.dumps(
                {"entry_ids": ids, "source": "external_fixture_not_extraction"},
                indent=2,
            )
        )
        return 0
    if args.command == "run":
        embedder = None
        started = time.monotonic()
        if args.mode == "hybrid":
            if not args.revision:
                raise SystemExit("--revision is required for a real hybrid run")
            embedder = SentenceTransformerEmbedder(MODEL_ID, revision=args.revision)
        memory = IndexedMemory(args.db, mode=args.mode, embedder=embedder)
        if embedder is not None:
            memory.rebuild_vectors()
        hits = memory.search(args.query, args.scope)
        payload = {
            "mode": args.mode,
            "query": args.query,
            "scope": args.scope,
            "elapsed_seconds": time.monotonic() - started,
            "model": MODEL_ID if embedder else None,
            "revision": args.revision if embedder else None,
            "dimension": embedder.dimension if embedder else None,
            "sentence_transformers_version": (
                importlib.metadata.version("sentence-transformers")
                if embedder
                else None
            ),
            "hits": [
                {"id": hit.entry.id, "rank": hit.rank, "score": hit.score}
                for hit in hits
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Memory validation\n\n```json\n" + json.dumps(rows, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
