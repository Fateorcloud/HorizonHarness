"""Bounded task-evidence extraction for indexed long-term memory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

from horizon_harness.llm import (
    LLMRequest,
    LLMResponse,
    Provider,
    TokenUsage,
    llm_message,
)
from horizon_harness.llm.retry import complete_with_tool_call_retry

from .indexed import MemoryType


@dataclass(frozen=True)
class MemoryEvidenceSource:
    ref: str
    run_id: str
    kind: str
    content: str


@dataclass(frozen=True)
class TaskMemoryEvidence:
    task_id: str
    run_id: str
    source_revision: int
    outcome: str
    scope: str
    task: str
    observed_at: str
    sources: tuple[MemoryEvidenceSource, ...]


@dataclass(frozen=True)
class ExtractedMemory:
    type: MemoryType
    content: str
    source_ref: str
    supersedes: str | None = None
    condition: str = ""
    method: str = ""
    outcome: str = ""
    conflict_group: str = ""


@dataclass(frozen=True)
class MemoryExtractionResult:
    candidates: tuple[ExtractedMemory, ...]
    usage: TokenUsage = TokenUsage()
    model: str = ""


class MemoryExtractor(Protocol):
    def extract(self, evidence: TaskMemoryEvidence) -> MemoryExtractionResult: ...


class ModelMemoryExtractor:
    """Use one bounded provider call to extract sourced memory candidates."""

    def __init__(
        self,
        provider: Provider,
        *,
        request_extra: Mapping[str, Any] | None = None,
        max_candidates: int = 10,
        max_evidence_chars: int = 30_000,
        complete_fn: Callable[
            [LLMRequest], LLMResponse
        ] = complete_with_tool_call_retry,
    ) -> None:
        if not 1 <= max_candidates <= 10:
            raise ValueError("max_candidates must be between 1 and 10")
        if max_evidence_chars <= 0:
            raise ValueError("max_evidence_chars must be greater than zero")
        self.provider = provider
        self.request_extra = dict(request_extra or {})
        self.max_candidates = max_candidates
        self.max_evidence_chars = max_evidence_chars
        self.complete_fn = complete_fn

    def extract(self, evidence: TaskMemoryEvidence) -> MemoryExtractionResult:
        request = LLMRequest(
            provider=self.provider,
            system_prompt=(
                "Extract durable, reusable repository memory only from the supplied "
                "task evidence. Return one JSON object with a candidates array. "
                "Each candidate has type (fact, decision, or experience), content, "
                "source_ref, and optional supersedes, condition, method, outcome, "
                "conflict_group. Experience requires condition, method, and the "
                "observed outcome. Do not infer success, credentials, hidden scores, "
                "or unfinished progress. Return at most "
                f"{self.max_candidates} candidates; an empty array is valid."
            ),
            messages=[llm_message("user", self._render_evidence(evidence))],
            max_tokens=2_048,
            extra=dict(self.request_extra),
        )
        response = self.complete_fn(request)
        payload = _json_object(response.text)
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("memory extractor response must contain a candidates list")
        if len(raw_candidates) > self.max_candidates:
            raise ValueError("memory extractor returned more candidates than requested")
        candidates = tuple(_candidate_from_json(row) for row in raw_candidates)
        return MemoryExtractionResult(candidates, response.usage, response.model)

    def _render_evidence(self, evidence: TaskMemoryEvidence) -> str:
        header = {
            "task_id": evidence.task_id,
            "run_id": evidence.run_id,
            "source_revision": evidence.source_revision,
            "outcome": evidence.outcome,
            "scope": evidence.scope,
            "task": evidence.task,
        }
        remaining = self.max_evidence_chars
        rendered_sources: list[dict[str, str]] = []
        for source in evidence.sources:
            if remaining <= 0:
                break
            content = source.content[:remaining]
            remaining -= len(content)
            rendered_sources.append(
                {
                    "ref": source.ref,
                    "run_id": source.run_id,
                    "kind": source.kind,
                    "content": content,
                }
            )
        return json.dumps(
            {"task": header, "sources": rendered_sources},
            ensure_ascii=False,
            sort_keys=True,
        )


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("memory extractor response must be a JSON object")
    return value


def _candidate_from_json(value: object) -> ExtractedMemory:
    if not isinstance(value, dict):
        raise ValueError("each extracted memory candidate must be an object")
    row = cast("dict[str, Any]", value)
    candidate_type = row.get("type")
    if candidate_type not in ("fact", "decision", "experience"):
        raise ValueError(f"invalid extracted memory type {candidate_type!r}")
    return ExtractedMemory(
        type=candidate_type,
        content=str(row.get("content", "")),
        source_ref=str(row.get("source_ref", "")),
        supersedes=_optional_text(row.get("supersedes")),
        condition=str(row.get("condition", "")),
        method=str(row.get("method", "")),
        outcome=str(row.get("outcome", "")),
        conflict_group=str(row.get("conflict_group", "")),
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
