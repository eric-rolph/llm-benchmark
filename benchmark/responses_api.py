"""Helpers for OpenAI Responses API-compatible backends."""
from __future__ import annotations

from typing import Any


def messages_to_responses_input(messages: list[dict]) -> list[dict]:
    """Convert chat-style messages into Responses API input items."""
    items: list[dict] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if isinstance(content, str):
            items.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            converted = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    converted.append({"type": "input_text", "text": str(part.get("text") or "")})
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url") or {}
                    if isinstance(image_url, dict) and image_url.get("url"):
                        converted.append({"type": "input_image", "image_url": str(image_url["url"])})
            items.append({"role": role, "content": converted or str(content)})
            continue
        items.append({"role": role, "content": str(content)})
    return items


def response_output_text(response: Any) -> str:
    """Extract visible text from a Responses API response object."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks)


def response_reasoning_preview(response: Any, limit: int = 200) -> str | None:
    """Extract an optional reasoning summary preview from a response."""
    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", None) or []:
            text = getattr(summary, "text", None)
            if text:
                chunks.append(str(text))
    if not chunks:
        return None
    return "\n".join(chunks)[:limit]


def response_usage_tokens(response: Any) -> tuple[int, int]:
    """Return (output_tokens, reasoning_tokens) from Responses or chat-shaped usage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    output_tokens = int(
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
        or 0
    )
    details = (
        getattr(usage, "output_tokens_details", None)
        or getattr(usage, "completion_tokens_details", None)
    )
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
    return output_tokens, reasoning_tokens
