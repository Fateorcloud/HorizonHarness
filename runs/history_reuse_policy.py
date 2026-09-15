"""Freeze the pilot's pre-intervention text without reverting runtime code."""

from dataclasses import replace
import re

from horizon_harness.compression import ToolCompactStrategy
from horizon_harness.context_view import CompressionDecision, ContextPolicy
from horizon_harness.messages import Message, TextBlock


class OriginalToolCompact(ToolCompactStrategy):
    """B/C keep precisely the original preview format, without source citations."""

    def __call__(
        self, active: list[tuple[int, Message]], agent_name: str
    ) -> CompressionDecision | None:
        decision = super().__call__(active, agent_name)
        if decision is None:
            return None
        replacement = decision.replacement
        content = tuple(
            TextBlock(
                re.sub(r" \[original transcript messages [0-9, -]+\]", "", block.text)
            )
            if isinstance(block, TextBlock)
            else block
            for block in replacement.content
        )
        return replace(decision, replacement=replace(replacement, content=content))


def policy_for(group: str) -> ContextPolicy:
    if group == "A":
        return ContextPolicy()
    strategy = ToolCompactStrategy if group == "D" else OriginalToolCompact
    return ContextPolicy(strategy=strategy(4000))
