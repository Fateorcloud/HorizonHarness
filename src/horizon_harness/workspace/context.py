"""Budgeted active-context projection backed by a Workspace record."""

from __future__ import annotations

from dataclasses import dataclass

from horizon_harness.context_view import (
    ContextPolicy,
    ContextView,
    build_context_view,
    estimate_context_tokens,
)
from horizon_harness.messages import Message, is_tool_result_message, runtime_message
from horizon_harness.state import State

from .models import WorkspaceRecord
from .store import WorkspaceStore


def render_workspace(record: WorkspaceRecord) -> str:
    done, total = record.progress
    lines = [
        "TASK WORKSPACE (authoritative task state; verifier results are separate)",
        f"task_id={record.task_id} revision={record.revision} status={record.status}",
        f"goal: {record.task}",
        "constraints: " + ("; ".join(record.constraints) or "none"),
        "acceptance: " + ("; ".join(record.acceptance) or "none"),
        f"progress: {done}/{total}",
    ]
    for step in record.steps:
        lines.append(
            f"step {step.id} [{step.status}]: {step.description}; evidence={list(step.evidence)}"
        )
    if record.next_action:
        lines.append(f"next: {record.next_action}")
    if record.blockers:
        lines.append("blockers: " + "; ".join(record.blockers))
    for decision in record.decisions:
        lines.append(f"decision: {decision.content} (why: {decision.rationale})")
    for artifact in record.artifacts:
        lines.append(f"artifact: {artifact.path} sha256={artifact.sha256}")
    for result in record.verifier_results:
        lines.append(
            f"verifier: {'PASS' if result.passed else 'FAIL'} {result.command}; "
            f"evidence={result.evidence_ref}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class WorkspaceContextBuilder:
    store: WorkspaceStore
    task_id: str
    max_input_tokens: int = 12_000
    fixed_overhead_tokens: int = 0
    fixed_overhead_tools: tuple[str, ...] = ()
    recent_turns: int = 2
    run_id: str = ""
    model_context_window: int | None = None
    output_reserve: int = 2_048
    safety_buffer: int = 1_024
    representation: str = "workspace"
    summary: str = ""
    summary_revision: int | None = None

    def __call__(
        self, agent_name: str, state: State, policy: ContextPolicy
    ) -> ContextView:
        record = self.store.load(self.task_id)
        if self.representation not in ("workspace", "summary", "combined"):
            raise ValueError(f"unknown context representation {self.representation!r}")
        if self.representation == "summary" and not self.summary.strip():
            raise ValueError("summary representation requires a non-empty summary")
        representation_text = (
            render_workspace(record)
            if self.representation == "workspace"
            else "CONTINUATION SUMMARY (non-authoritative)\n" + self.summary
        )
        if self.representation == "combined":
            if not self.summary.strip() or self.summary_revision is None:
                raise ValueError(
                    "combined context requires summary and source revision"
                )
            if self.summary_revision > record.revision:
                raise ValueError("summary revision is newer than current workspace")
            representation_text = "\n".join(
                (
                    f"CURRENT TASK STATE revision={record.revision} status={record.status}",
                    "constraints: " + "; ".join(record.constraints),
                    "acceptance: " + "; ".join(record.acceptance),
                    "next: " + record.next_action,
                    "Read workspace_read for steps, decisions, artifacts and blockers.",
                    f"HANDOFF SUMMARY source_revision={self.summary_revision} "
                    f"stale={self.summary_revision != record.revision}",
                    "Summary is historical context, not current authority. Current task "
                    "and constraints take precedence. On conflict read workspace_read "
                    "and original evidence before acting.",
                    self.summary,
                )
            )
        workspace = runtime_message(
            representation_text,
            sender=self.representation,
            target=agent_name,
            kind="context",
        )
        visible = [message for message in state.messages if policy.is_visible(message)]
        task_messages = [message for message in visible if message.kind == "task"][:1]
        memory_messages = [
            message
            for message in visible
            if message.kind == "context" and message.sender == "memory"
        ][-1:]
        message_refs = [*record.checkpoint.message_refs]
        message_refs.extend(
            f"{self.run_id}:{index}"
            for index in range(len(visible) - len(message_refs))
        )
        recent_turns, archive_turns = _complete_turns(visible, self.recent_turns)
        recent = [message for turn in recent_turns for _, message in turn]
        selected: list[Message] = [*task_messages, *memory_messages, workspace, *recent]
        required_tokens = estimate_context_tokens(
            [*task_messages, *memory_messages, workspace], allow_usage_baseline=False
        )
        budget = self.max_input_tokens - self.fixed_overhead_tokens
        if required_tokens > budget:
            raise ValueError(
                "workspace goal, constraints, and current state exceed the active "
                f"context budget ({required_tokens} > {budget} estimated tokens)"
            )
        while (
            estimate_context_tokens(selected, allow_usage_baseline=False) > budget
            and len(recent_turns) > 1
        ):
            archive_turns.append(recent_turns.pop(0))
            recent = [message for turn in recent_turns for _, message in turn]
            selected = [*task_messages, *memory_messages, workspace, *recent]
        selected_tokens = estimate_context_tokens(selected, allow_usage_baseline=False)
        if selected_tokens > budget:
            raise ValueError(
                "workspace required state and latest complete turn exceed the active "
                f"context budget ({selected_tokens} > {budget} estimated tokens)"
            )
        recent_indices = [index for turn in recent_turns for index, _ in turn]
        archive_indices = [index for turn in archive_turns for index, _ in turn]
        view = build_context_view(agent_name, selected, policy=policy)
        return ContextView(
            agent=view.agent,
            messages=view.messages,
            stats=view.stats,
            metadata={
                "builder": self.representation,
                "checkpoint_revision": record.revision,
                "summary_revision": self.summary_revision,
                "included_message_refs": [
                    message_refs[index] for index in recent_indices
                ],
                "archive_refs": [message_refs[index] for index in archive_indices],
                "fixed_overhead_tokens": self.fixed_overhead_tokens,
                "fixed_overhead_tools": list(self.fixed_overhead_tools),
                "max_input_tokens": self.max_input_tokens,
                "active_message_budget": budget,
                "model_context_window": self.model_context_window,
                "output_reserve": self.output_reserve,
                "safety_buffer": self.safety_buffer,
                "trim_reason": "old_complete_turns" if archive_indices else "",
            },
        )


def _complete_turns(
    messages: list[Message], keep: int
) -> tuple[list[list[tuple[int, Message]]], list[list[tuple[int, Message]]]]:
    turns: list[list[tuple[int, Message]]] = []
    current: list[tuple[int, Message]] = []
    for index, message in enumerate(messages):
        if message.role == "assistant":
            if current:
                turns.append(current)
            current = [(index, message)]
        elif current and is_tool_result_message(message):
            current.append((index, message))
    if current:
        turns.append(current)
    kept = turns[-keep:] if keep else []
    old = turns[: max(0, len(turns) - keep)]
    return list(kept), list(old)
