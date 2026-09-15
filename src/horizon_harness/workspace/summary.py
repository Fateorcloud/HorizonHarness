"""One-shot bounded continuation summary used by the formal S baseline."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horizon_harness.llm import (
    LLMRequest,
    LLMResponse,
    Provider,
    TokenUsage,
    llm_message,
)
from horizon_harness.llm.retry import complete_with_tool_call_retry
from horizon_harness.messages import Message

from .codec import encode_message
from .models import WorkspaceRecord
from .tools import apply_workspace_update


@dataclass(frozen=True)
class SummaryResult:
    text: str
    usage: TokenUsage
    model: str


@dataclass(frozen=True)
class WorkspaceProjectionResult:
    record: WorkspaceRecord
    usage: TokenUsage
    model: str


class ModelTaskSummarizer:
    def __init__(
        self,
        provider: Provider,
        *,
        request_extra: Mapping[str, Any] | None = None,
        max_evidence_chars: int = 30_000,
        complete_fn: Callable[
            [LLMRequest], LLMResponse
        ] = complete_with_tool_call_retry,
    ) -> None:
        self.provider = provider
        self.request_extra = dict(request_extra or {})
        self.max_evidence_chars = max_evidence_chars
        self.complete_fn = complete_fn

    def summarize(self, task: str, messages: Sequence[Message]) -> SummaryResult:
        transcript = json.dumps(
            [encode_message(message) for message in messages],
            ensure_ascii=False,
            sort_keys=True,
        )[: self.max_evidence_chars]
        request = LLMRequest(
            provider=self.provider,
            system_prompt=(
                "Create a factual continuation summary from the task transcript. "
                "Preserve the goal, constraints, completed actions, tool observations, "
                "changed files, unresolved work, and next action. Do not claim verifier "
                "success or invent facts. Return summary text only."
            ),
            messages=[
                llm_message("user", f"TASK:\n{task}\n\nTRANSCRIPT:\n{transcript}")
            ],
            max_tokens=2_048,
            extra=dict(self.request_extra),
        )
        response = self.complete_fn(request)
        text = response.text.strip()
        if not text:
            raise ValueError("continuation summarizer returned empty text")
        return SummaryResult(text, response.usage, response.model)


class ModelWorkspaceProjector:
    """Create one validated structured W representation from a plain prefix."""

    def __init__(
        self,
        provider: Provider,
        *,
        request_extra: Mapping[str, Any] | None = None,
        max_evidence_chars: int = 30_000,
        complete_fn: Callable[
            [LLMRequest], LLMResponse
        ] = complete_with_tool_call_retry,
    ) -> None:
        self.provider = provider
        self.request_extra = dict(request_extra or {})
        self.max_evidence_chars = max_evidence_chars
        self.complete_fn = complete_fn

    def project(
        self,
        record: WorkspaceRecord,
        messages: Sequence[Message],
        *,
        code_dir: Path,
    ) -> WorkspaceProjectionResult:
        transcript = json.dumps(
            [encode_message(message) for message in messages],
            ensure_ascii=False,
            sort_keys=True,
        )[: self.max_evidence_chars]
        request = LLMRequest(
            provider=self.provider,
            system_prompt=(
                "Create a structured task continuation state from the transcript. "
                "Return one JSON object with steps, decisions, artifacts, blockers, "
                "next_action, and status='paused'. Steps need stable id, description, "
                "status (pending, in_progress, done, blocked), and evidence refs. Decisions need content, rationale, step_ids, "
                "and evidence. Artifacts must be existing relative paths and may include "
                "step_ids; never claim verifier results. Omit temporary files outside "
                "the repository (including /tmp); they are not persistent artifacts."
            ),
            messages=[
                llm_message(
                    "user",
                    json.dumps(
                        {
                            "task": record.task,
                            "constraints": record.constraints,
                            "acceptance": record.acceptance,
                            "transcript": transcript,
                        },
                        ensure_ascii=False,
                    ),
                )
            ],
            max_tokens=2_048,
            extra=dict(self.request_extra),
        )
        response = self.complete_fn(request)
        value = _parse_json_object(response.text)
        if not isinstance(value, dict):
            raise ValueError("workspace projector response must be a JSON object")
        args = dict(value)
        artifacts = []
        for artifact in args.get("artifacts", []):
            path = Path(str(artifact.get("path", "")))
            if path.is_absolute():
                args.setdefault("blockers", []).append(
                    f"External artifact not persisted; recreate if needed: {path}"
                )
            else:
                artifacts.append(artifact)
        args["artifacts"] = artifacts
        for step in args.get("steps", []):
            if step.get("status") == "completed":
                step["status"] = "done"
        args["expected_revision"] = record.revision
        args["status"] = "paused"
        revised = apply_workspace_update(record, args, code_dir=code_dir)
        return WorkspaceProjectionResult(revised, response.usage, response.model)


def _parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[0] not in ("```", "```json"):
            raise ValueError("workspace projector returned an invalid JSON fence")
        value = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("workspace projector response must be a JSON object")
    return parsed
