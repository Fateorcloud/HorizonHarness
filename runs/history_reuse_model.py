"""Single-attempt model boundary and per-run ledger for the pilot only."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable

from openai import OpenAI

from horizon_harness.core import Agent
from horizon_harness.llm.adapters.openai_chat import (
    to_openai_chat_messages,
    to_openai_chat_tools,
)
from horizon_harness.llm.bridge import messages_to_llm_messages, tool_to_llm_tool
from horizon_harness.messages import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    TokenUsage,
    assistant_message,
)


class BudgetExhausted(RuntimeError):
    """No further request is allowed for this run."""


@dataclass
class Ledger:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    started: float = field(default_factory=time.monotonic)
    requests: list[dict[str, Any]] = field(default_factory=list)
    max_input_tokens: int = 64000
    request_timeout_seconds: float = 60.0

    def reserve(self, wire: dict[str, Any]) -> int:
        # Conservative byte proxy, including tools and a framing allowance.
        # Not the provider tokenizer: record the proxy separately from usage.
        proxy = len(json.dumps(wire, ensure_ascii=False).encode("utf-8")) + 4096
        if (
            self.calls >= 12
            or self.input_tokens + proxy > self.max_input_tokens
            or self.output_tokens >= 8000
            or time.monotonic() - self.started >= 600
        ):
            raise BudgetExhausted("per-run request/token/time limit")
        self.calls += 1
        return min(2048, 8000 - self.output_tokens)


def bounded_generate(
    agent: Agent, client: OpenAI, ledger: Ledger
) -> Callable[[list[Message]], Message]:
    """Keep runtime assembly, replacing only its retrying network closure."""
    provider = agent.llm_provider
    assert provider is not None
    tools = to_openai_chat_tools([tool_to_llm_tool(t) for t in agent.tools])

    def generate(visible: list[Message]) -> Message:
        wire = {
            "model": provider.model,
            "messages": to_openai_chat_messages(
                messages_to_llm_messages(visible),
                system_prompt=agent.system_prompt,
            ),
            "tools": tools,
        }
        cap = ledger.reserve(wire)
        record: dict[str, Any] = {
            "request": {**wire, "max_tokens": cap},
            "input_byte_proxy": len(
                json.dumps(wire, ensure_ascii=False).encode("utf-8")
            )
            + 4096,
        }
        ledger.requests.append(record)
        timeout = max(
            1,
            min(
                ledger.request_timeout_seconds,
                600 - (time.monotonic() - ledger.started),
            ),
        )
        record["timeout_seconds"] = timeout
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                **wire,
                max_tokens=cap,
                timeout=timeout,
            )
        except Exception as exc:
            record["error"] = {
                "type": type(exc).__name__,
                "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
                "http_status": getattr(exc, "status_code", None),
                "request_id": getattr(exc, "request_id", None),
            }
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
        record["response"] = response.model_dump()
        if response.usage is None:
            raise BudgetExhausted(
                "provider usage unavailable; stop, do not price as zero"
            )
        usage = response.usage
        ledger.input_tokens += usage.prompt_tokens
        ledger.output_tokens += usage.completion_tokens
        if not response.choices:
            raise RuntimeError("provider returned no choices; no retry")
        choice = response.choices[0]
        message = choice.message
        blocks = []
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            blocks.append(
                ThinkingBlock(text=reasoning, source_field="reasoning_content")
            )
        if message.content:
            blocks.append(TextBlock(message.content))
        for call in message.tool_calls or []:
            blocks.append(
                ToolCallBlock(
                    call.id,
                    call.function.name,
                    json.loads(call.function.arguments),
                )
            )
        cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        return assistant_message(
            blocks,
            sender=agent.name,
            target="user",
            kind="final" if choice.finish_reason == "stop" else "step",
            model=response.model,
            usage=TokenUsage.from_inclusive_input(
                total_input=usage.prompt_tokens,
                output=usage.completion_tokens,
                cached_read=cached,
            ),
        )

    return generate
