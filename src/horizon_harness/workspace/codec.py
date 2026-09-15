"""Lossless JSON codec for the text-oriented project message protocol."""

from __future__ import annotations

from typing import Any
import json
from horizon_harness.tools import ToolResult

from horizon_harness.messages import (
    AssistantMessage,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    assistant_message,
    runtime_message,
    tool_results_message,
    user_message,
)


def encode_tool_result(result: ToolResult) -> dict[str, Any]:
    return {
        "result_version": 1,
        "content": [_encode_block(block) for block in result.content],
        "details": json.loads(json.dumps(result.details, allow_nan=False)),
        "is_error": result.is_error,
        "terminate": result.terminate,
    }


def decode_tool_result(value: dict[str, Any]) -> ToolResult:
    if value.get("result_version") != 1:
        raise ValueError("unsupported tool result version")
    blocks = tuple(_decode_block(block) for block in value["content"])
    if any(not isinstance(block, (TextBlock, ImageBlock)) for block in blocks):
        raise ValueError("tool result content must contain text or images")
    if (
        type(value.get("is_error")) is not bool
        or type(value.get("terminate")) is not bool
    ):
        raise ValueError("tool result flags must be booleans")
    return ToolResult(
        content=blocks,
        details=json.loads(json.dumps(value.get("details"))),
        is_error=value["is_error"],
        terminate=value["terminate"],
    )


def encode_message(message: Message) -> dict[str, Any]:
    return {
        "message_version": 1,
        "sidecar": json.loads(json.dumps(message.sidecar, allow_nan=False)),
        "role": message.role,
        "sender": message.sender,
        "target": message.target,
        "kind": message.kind,
        "content": [_encode_block(block) for block in message.content],
        "usage": (
            vars(message.usage)
            if isinstance(message, AssistantMessage) and message.usage
            else None
        ),
        "model": message.model if isinstance(message, AssistantMessage) else "",
    }


def decode_message(value: dict[str, Any]) -> Message:
    if value.get("message_version", 1) != 1:
        raise ValueError("unsupported canonical message version")
    if value.get("role") not in ("assistant", "system", "user"):
        raise ValueError("unsupported canonical message role")
    sidecar = value.get("sidecar", {})
    if not isinstance(sidecar, dict):
        raise ValueError("canonical message sidecar must be an object")
    content = tuple(_decode_block(block) for block in value.get("content", ()))
    common = {
        "sender": value["sender"],
        "target": value["target"],
        "kind": value["kind"],
        "sidecar": json.loads(json.dumps(sidecar, allow_nan=False)),
    }
    if value["role"] == "assistant":
        usage = value.get("usage")
        return assistant_message(
            content,
            **common,
            usage=TokenUsage(**usage) if usage else None,
            model=value.get("model", ""),
        )
    if value["role"] == "system":
        return runtime_message(content, **common)
    if all(isinstance(block, ToolResultBlock) for block in content) and content:
        return tool_results_message(content, **common)
    return user_message(content, **common)


def _encode_block(block: Any) -> dict[str, Any]:
    row = {"kind": block.kind}
    if isinstance(block, TextBlock):
        row["text"] = block.text
    elif isinstance(block, ImageBlock):
        row.update(data=block.data, mime_type=block.mime_type)
    elif isinstance(block, ThinkingBlock):
        row.update(
            text=block.text,
            signature=block.signature,
            redacted=block.redacted,
            source_field=block.source_field,
        )
    elif isinstance(block, ToolCallBlock):
        row.update(id=block.id, name=block.name, arguments=dict(block.arguments))
    elif isinstance(block, ToolResultBlock):
        row.update(
            tool_call_id=block.tool_call_id,
            tool_name=block.tool_name,
            is_error=block.is_error,
            content=[_encode_block(item) for item in block.content],
        )
    else:
        raise TypeError(f"unsupported message block {type(block)!r}")
    return row


def _decode_block(value: dict[str, Any]) -> Any:
    kind = value["kind"]
    if kind == "text":
        return TextBlock(value["text"])
    if kind == "image":
        return ImageBlock(value["data"], value["mime_type"])
    if kind == "thinking":
        return ThinkingBlock(
            text=value.get("text", ""),
            signature=value.get("signature"),
            redacted=bool(value.get("redacted", False)),
            source_field=value.get("source_field"),
        )
    if kind == "tool_call":
        return ToolCallBlock(value["id"], value["name"], value.get("arguments", {}))
    if kind == "tool_result":
        return ToolResultBlock(
            tool_call_id=value["tool_call_id"],
            tool_name=value["tool_name"],
            is_error=bool(value.get("is_error", False)),
            content=tuple(_decode_block(item) for item in value.get("content", ())),
        )
    raise ValueError(f"unsupported content block kind {kind!r}")
