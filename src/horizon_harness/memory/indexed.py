"""SQLite-backed sourced memory with keyword and optional hybrid retrieval."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sqlite3
import struct
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Protocol, Sequence

from horizon_harness.messages import Message
from horizon_harness.tools import AgentTool, text_result

from .base import Memory, MemoryContext, memory_context_message

if TYPE_CHECKING:
    from .extraction import (
        MemoryExtractionResult,
        MemoryExtractor,
        TaskMemoryEvidence,
    )


MemoryType = Literal["fact", "decision", "experience"]
SearchMode = Literal["keyword", "hybrid"]


class Embedder(Protocol):
    model_id: str
    revision: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class MemoryCandidate:
    type: MemoryType
    content: str
    scope: str
    task_id: str
    run_id: str
    source_ref: str
    observed_at: str
    source_revision: int
    supersedes: str | None = None
    condition: str = ""
    method: str = ""
    outcome: str = ""
    conflict_group: str = ""


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    type: MemoryType
    content: str
    scope: str
    task_id: str
    run_id: str
    source_ref: str
    observed_at: str
    created_at: float
    source_revision: int
    supersedes: str | None
    status: str
    condition: str
    method: str
    outcome: str
    conflict_group: str


@dataclass(frozen=True)
class SearchHit:
    entry: MemoryEntry
    rank: int
    score: float
    retrieval: SearchMode
    conflict: bool


class SentenceTransformerEmbedder:
    """Lazy CPU embedder; importing base HorizonHarness stays lightweight."""

    def __init__(
        self,
        model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        *,
        revision: str,
    ) -> None:
        if not revision:
            raise ValueError("an exact embedding model revision is required")
        self.model_id = model_id
        self.revision = revision
        self.dimension = 0
        self._model = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            sentence_transformers = importlib.import_module("sentence_transformers")
            SentenceTransformer = sentence_transformers.SentenceTransformer
            self._model = SentenceTransformer(
                self.model_id, revision=self.revision, device="cpu"
            )
            dimension = self._model.get_embedding_dimension()
            if dimension is None:
                raise RuntimeError("embedding model did not report its dimension")
            self.dimension = int(dimension)
        values = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(item) for item in row] for row in values]


class IndexedMemory(Memory):
    """Small sourced entry store; no task-end action occurs on SESSION_END."""

    def __init__(
        self,
        path: str | Path,
        *,
        mode: SearchMode = "keyword",
        embedder: Embedder | None = None,
    ) -> None:
        if mode == "hybrid" and embedder is None:
            raise ValueError(
                "hybrid mode requires an embedder; use keyword for explicit fallback"
            )
        self.path = Path(path).expanduser().resolve()
        self.mode = mode
        self.embedder = embedder
        self._initialize()

    def initial(self, ctx: MemoryContext) -> tuple[Message, ...]:
        return (
            memory_context_message(
                "Long-term memory is available through memory_search and memory_read. "
                "Search only when prior task evidence may help; current code and user "
                "instructions take precedence, and source scope must match.",
                target=ctx.agent,
            ),
        )

    def tools(self, ctx: MemoryContext) -> tuple[AgentTool, ...]:
        scope = str(ctx.data.get("scope", ""))
        return (self._search_tool(scope), self._read_tool(scope))

    def finish(self, ctx: MemoryContext) -> None:
        # A generic SESSION_END includes pauses and budget stops. Extraction is
        # deliberately explicit through finish_task.
        del ctx

    def register_sources(self, task_id: str, run_id: str, refs: Iterable[str]) -> None:
        with self._connect() as db:
            db.executemany(
                "INSERT OR IGNORE INTO sources(task_id, run_id, source_ref) VALUES (?, ?, ?)",
                ((task_id, run_id, ref) for ref in refs),
            )

    def finish_task(
        self,
        *,
        task_id: str,
        source_revision: int,
        outcome: str,
        candidates: Sequence[MemoryCandidate],
    ) -> tuple[str, ...]:
        if outcome not in ("completed", "failed"):
            raise ValueError(
                "memory extraction requires explicit completed or failed outcome"
            )
        if len(candidates) > 10:
            raise ValueError(
                "at most 10 memory candidates may be extracted from one task"
            )
        with self._connect() as db:
            prior = db.execute(
                "SELECT entry_ids FROM finishes WHERE task_id=? AND source_revision=?",
                (task_id, source_revision),
            ).fetchone()
            if prior:
                return tuple(json.loads(prior[0]))
            ids = tuple(
                self._insert_candidate(db, candidate) for candidate in candidates
            )
            db.execute(
                "INSERT INTO finishes(task_id, source_revision, outcome, entry_ids) VALUES (?, ?, ?, ?)",
                (task_id, source_revision, outcome, json.dumps(ids)),
            )
            return ids

    def extract_task(
        self,
        evidence: TaskMemoryEvidence,
        extractor: MemoryExtractor,
    ) -> tuple[tuple[str, ...], MemoryExtractionResult]:
        """Extract from bounded evidence, validate source ownership, and persist."""

        if evidence.outcome not in ("completed", "failed"):
            raise ValueError(
                "memory extraction requires explicit completed or failed outcome"
            )
        if not evidence.scope.strip():
            raise ValueError("memory extraction scope must not be empty")
        refs = tuple(source.ref for source in evidence.sources)
        if len(refs) != len(set(refs)):
            raise ValueError("memory evidence source refs must be unique")
        for source in evidence.sources:
            self.register_sources(evidence.task_id, source.run_id, (source.ref,))
        result = extractor.extract(evidence)
        sources = {source.ref: source for source in evidence.sources}
        candidates = tuple(
            MemoryCandidate(
                type=item.type,
                content=item.content,
                scope=evidence.scope,
                task_id=evidence.task_id,
                run_id=(
                    sources[item.source_ref].run_id
                    if item.source_ref in sources
                    else evidence.run_id
                ),
                source_ref=item.source_ref,
                observed_at=evidence.observed_at,
                source_revision=evidence.source_revision,
                supersedes=item.supersedes,
                condition=item.condition,
                method=item.method,
                outcome=item.outcome,
                conflict_group=item.conflict_group,
            )
            for item in result.candidates
        )
        ids = self.finish_task(
            task_id=evidence.task_id,
            source_revision=evidence.source_revision,
            outcome=evidence.outcome,
            candidates=candidates,
        )
        return ids, result

    def search(
        self,
        query: str,
        scope: str,
        limit: int = 5,
        *,
        mode: SearchMode | None = None,
        max_tokens: int = 2_000,
    ) -> tuple[SearchHit, ...]:
        if not query.strip() or not scope.strip():
            return ()
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        selected_mode = mode or self.mode
        with self._connect() as db:
            keyword = self._keyword_rank(db, query, scope, 20)
            if selected_mode == "keyword":
                ranked = [(entry_id, score) for entry_id, score in keyword]
            else:
                if self.embedder is None:
                    raise RuntimeError(
                        "hybrid retrieval cannot silently fall back without an embedder"
                    )
                semantic = self._semantic_rank(db, query, scope, 20)
                ranked = _rrf(keyword, semantic)
            hits: list[SearchHit] = []
            used = 0
            for rank, (entry_id, score) in enumerate(ranked, 1):
                entry = self._entry(db, entry_id)
                if entry is None:
                    continue
                estimated = (len(entry.content) + len(entry.condition) + 2) // 3
                if hits and used + estimated > max_tokens:
                    break
                used += estimated
                conflict = bool(
                    entry.conflict_group
                    and db.execute(
                        "SELECT 1 FROM entries WHERE scope=? AND conflict_group=? "
                        "AND status='active' AND id<>? LIMIT 1",
                        (scope, entry.conflict_group, entry.id),
                    ).fetchone()
                )
                hits.append(SearchHit(entry, rank, score, selected_mode, conflict))
                if len(hits) >= limit:
                    break
            return tuple(hits)

    def get(self, ids: Sequence[str], scope: str) -> tuple[MemoryEntry, ...]:
        with self._connect() as db:
            entries = [self._entry(db, entry_id) for entry_id in ids]
        return tuple(
            entry for entry in entries if entry is not None and entry.scope == scope
        )

    def rebuild_vectors(self) -> int:
        if self.embedder is None:
            raise RuntimeError("an embedder is required to rebuild vectors")
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, content, condition, method, outcome FROM entries"
            ).fetchall()
            missing: list[tuple[str, str]] = []
            for row in rows:
                text = _embedding_text(row[1], row[2], row[3], row[4])
                digest = _embedding_hash(
                    self.embedder.model_id, self.embedder.revision, text
                )
                cached = db.execute(
                    "SELECT 1 FROM vectors WHERE entry_id=? AND content_hash=? "
                    "AND model_id=? AND revision=?",
                    (row[0], digest, self.embedder.model_id, self.embedder.revision),
                ).fetchone()
                if not cached:
                    missing.append((str(row[0]), text))
            vectors = (
                self.embedder.encode([text for _, text in missing]) if missing else []
            )
            for (entry_id, text), vector in zip(missing, vectors):
                self._write_vector(db, entry_id, text, vector)
            return len(vectors)

    def _insert_candidate(
        self, db: sqlite3.Connection, candidate: MemoryCandidate
    ) -> str:
        self._validate_candidate(db, candidate)
        entry_id = uuid.uuid4().hex
        status = "active"
        if candidate.supersedes:
            db.execute(
                "UPDATE entries SET status='superseded' WHERE id=?",
                (candidate.supersedes,),
            )
        db.execute(
            """INSERT INTO entries(
                id,type,content,scope,task_id,run_id,source_ref,observed_at,
                created_at,source_revision,supersedes,status,condition,method,
                outcome,conflict_group,search_text
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                entry_id,
                candidate.type,
                candidate.content,
                candidate.scope,
                candidate.task_id,
                candidate.run_id,
                candidate.source_ref,
                candidate.observed_at,
                time.time(),
                candidate.source_revision,
                candidate.supersedes,
                status,
                candidate.condition,
                candidate.method,
                candidate.outcome,
                candidate.conflict_group,
                tokenize(" ".join((candidate.content, candidate.condition))),
            ),
        )
        db.execute(
            "INSERT INTO entries_fts(rowid, search_text) VALUES ((SELECT rowid FROM entries WHERE id=?), ?)",
            (entry_id, tokenize(" ".join((candidate.content, candidate.condition)))),
        )
        if self.embedder is not None:
            text = _embedding_text(
                candidate.content,
                candidate.condition,
                candidate.method,
                candidate.outcome,
            )
            vector = self.embedder.encode([text])[0]
            self._write_vector(db, entry_id, text, vector)
        return entry_id

    def _validate_candidate(
        self, db: sqlite3.Connection, candidate: MemoryCandidate
    ) -> None:
        if candidate.type not in ("fact", "decision", "experience"):
            raise ValueError(f"invalid memory type {candidate.type!r}")
        if not candidate.content.strip() or len(candidate.content) > 1_000:
            raise ValueError("memory content must contain 1..1000 characters")
        sensitive = re.search(
            r"(?i)(?:api[_-]?key|auth[_-]?token|password)\s*[:=]\s*\S+|"
            r"-----BEGIN [A-Z ]+PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{12,}",
            candidate.content,
        )
        if sensitive:
            raise ValueError("memory candidate appears to contain a credential")
        if not candidate.scope.strip():
            raise ValueError("memory scope must not be empty")
        if candidate.type == "experience" and not all(
            (
                candidate.condition.strip(),
                candidate.method.strip(),
                candidate.outcome.strip(),
            )
        ):
            raise ValueError(
                "experience requires condition, method, and observed outcome"
            )
        source = db.execute(
            "SELECT 1 FROM sources WHERE task_id=? AND run_id=? AND source_ref=?",
            (candidate.task_id, candidate.run_id, candidate.source_ref),
        ).fetchone()
        if not source:
            raise ValueError(
                f"source reference does not belong to task/run: {candidate.source_ref!r}"
            )
        if candidate.supersedes:
            old = self._entry(db, candidate.supersedes)
            if old is None or old.status != "active":
                raise ValueError("supersedes must reference an active entry")
            if old.scope != candidate.scope:
                raise ValueError("an entry cannot supersede memory from another scope")

    def _keyword_rank(
        self, db: sqlite3.Connection, query: str, scope: str, limit: int
    ) -> list[tuple[str, float]]:
        tokens = tokenize(query).split()
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        rows = db.execute(
            """SELECT e.id, bm25(entries_fts) FROM entries_fts
               JOIN entries e ON e.rowid=entries_fts.rowid
               WHERE entries_fts MATCH ? AND e.scope=? AND e.status='active'
               ORDER BY bm25(entries_fts), e.id LIMIT ?""",
            (expression, scope, limit),
        ).fetchall()
        return [(row[0], -float(row[1])) for row in rows]

    def _semantic_rank(
        self, db: sqlite3.Connection, query: str, scope: str, limit: int
    ) -> list[tuple[str, float]]:
        assert self.embedder is not None
        query_vector = self.embedder.encode([query])[0]
        rows = db.execute(
            """SELECT e.id,v.model_id,v.revision,v.dimension,v.vector
               FROM entries e JOIN vectors v ON v.entry_id=e.id
               WHERE e.scope=? AND e.status='active'""",
            (scope,),
        ).fetchall()
        scored: list[tuple[str, float]] = []
        for entry_id, model_id, revision, dimension, blob in rows:
            if model_id != self.embedder.model_id or revision != self.embedder.revision:
                raise RuntimeError(
                    "embedding model revision mismatch; rebuild the vector index"
                )
            if int(dimension) != len(query_vector):
                raise RuntimeError(
                    "embedding dimension mismatch; rebuild the vector index"
                )
            vector = list(struct.unpack(f"<{dimension}f", blob))
            scored.append((entry_id, _cosine(query_vector, vector)))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]

    def _write_vector(
        self, db: sqlite3.Connection, entry_id: str, text: str, vector: Sequence[float]
    ) -> None:
        assert self.embedder is not None
        if not vector:
            raise ValueError("embedder returned an empty vector")
        dimension = len(vector)
        if self.embedder.dimension not in (0, dimension):
            raise ValueError(
                f"embedder dimension mismatch: declared {self.embedder.dimension}, got {dimension}"
            )
        digest = _embedding_hash(self.embedder.model_id, self.embedder.revision, text)
        db.execute(
            """INSERT OR REPLACE INTO vectors(
               entry_id,content_hash,model_id,revision,dimension,vector
               ) VALUES (?,?,?,?,?,?)""",
            (
                entry_id,
                digest,
                self.embedder.model_id,
                self.embedder.revision,
                dimension,
                struct.pack(f"<{dimension}f", *vector),
            ),
        )

    def _entry(self, db: sqlite3.Connection, entry_id: str) -> MemoryEntry | None:
        row = db.execute(
            """SELECT id,type,content,scope,task_id,run_id,source_ref,observed_at,
               created_at,source_revision,supersedes,status,condition,method,
               outcome,conflict_group FROM entries WHERE id=?""",
            (entry_id,),
        ).fetchone()
        return MemoryEntry(*row) if row else None

    def _search_tool(self, scope: str) -> AgentTool:
        def execute(call_id, args, abort, on_update):
            del call_id, abort, on_update
            try:
                hits = self.search(str(args["query"]), scope, int(args.get("limit", 5)))
                return text_result(
                    json.dumps([_hit_dict(hit) for hit in hits], ensure_ascii=False)
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                return text_result(str(exc), is_error=True)

        return AgentTool(
            name="memory_search",
            description="Search sourced long-term memory within the current repository scope.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            execute=execute,
            execution_mode="sequential",
        )

    def _read_tool(self, scope: str) -> AgentTool:
        def execute(call_id, args, abort, on_update):
            del call_id, abort, on_update
            try:
                entries = self.get(list(args["ids"]), scope)
                return text_result(
                    json.dumps([asdict(entry) for entry in entries], ensure_ascii=False)
                )
            except (KeyError, TypeError, ValueError) as exc:
                return text_result(str(exc), is_error=True)

        return AgentTool(
            name="memory_read",
            description="Read full memory entries and source references by id in current scope.",
            parameters={
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}, "maxItems": 5}
                },
                "required": ["ids"],
                "additionalProperties": False,
            },
            execute=execute,
            execution_mode="sequential",
        )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources(
                    task_id TEXT NOT NULL, run_id TEXT NOT NULL, source_ref TEXT NOT NULL,
                    PRIMARY KEY(task_id,run_id,source_ref)
                );
                CREATE TABLE IF NOT EXISTS entries(
                    id TEXT PRIMARY KEY, type TEXT NOT NULL, content TEXT NOT NULL,
                    scope TEXT NOT NULL, task_id TEXT NOT NULL, run_id TEXT NOT NULL,
                    source_ref TEXT NOT NULL, observed_at TEXT NOT NULL,
                    created_at REAL NOT NULL, source_revision INTEGER NOT NULL,
                    supersedes TEXT, status TEXT NOT NULL, condition TEXT NOT NULL,
                    method TEXT NOT NULL, outcome TEXT NOT NULL,
                    conflict_group TEXT NOT NULL, search_text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(search_text);
                CREATE TABLE IF NOT EXISTS vectors(
                    entry_id TEXT PRIMARY KEY REFERENCES entries(id),
                    content_hash TEXT NOT NULL, model_id TEXT NOT NULL,
                    revision TEXT NOT NULL, dimension INTEGER NOT NULL, vector BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS finishes(
                    task_id TEXT NOT NULL, source_revision INTEGER NOT NULL,
                    outcome TEXT NOT NULL, entry_ids TEXT NOT NULL,
                    PRIMARY KEY(task_id,source_revision)
                );
                """
            )


_WORD = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]+")


def tokenize(text: str) -> str:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ")
    tokens: list[str] = []
    for match in _WORD.findall(expanded):
        lowered = match.lower()
        if re.fullmatch(r"[\u3400-\u9fff]+", lowered):
            tokens.extend(
                lowered
                if len(lowered) == 1
                else (lowered[index : index + 2] for index in range(len(lowered) - 1))
            )
        else:
            tokens.append(lowered)
    return " ".join(tokens)


def _embedding_text(content: str, condition: str, method: str, outcome: str) -> str:
    return "\n".join(part for part in (content, condition, method, outcome) if part)


def _embedding_hash(model_id: str, revision: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\0{revision}\0{text}".encode()).hexdigest()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(
        sum(x * x for x in right)
    )
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def _rrf(
    keyword: Sequence[tuple[str, float]], semantic: Sequence[tuple[str, float]]
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in (keyword, semantic):
        for rank, (entry_id, _) in enumerate(ranked, 1):
            scores[entry_id] = scores.get(entry_id, 0.0) + 1.0 / (60 + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _hit_dict(hit: SearchHit) -> dict[str, object]:
    entry = hit.entry
    return {
        "id": entry.id,
        "type": entry.type,
        "content": entry.content,
        "scope": entry.scope,
        "source": {
            "task_id": entry.task_id,
            "run_id": entry.run_id,
            "ref": entry.source_ref,
            "revision": entry.source_revision,
        },
        "supersedes": entry.supersedes,
        "conflict": hit.conflict,
        "rank": hit.rank,
        "retrieval": hit.retrieval,
    }
