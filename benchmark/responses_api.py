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
    metadata = response_usage_metadata(response)
    return int(metadata["completion_tokens"] or 0), int(metadata["reasoning_tokens"] or 0)


def response_usage_metadata(response: Any) -> dict:
    """Return comparable token/cost metadata from Responses or chat-shaped usage."""
    usage = _get(response, "usage")
    return usage_metadata(usage)


def usage_metadata(usage: Any) -> dict:
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "api_cost": None,
        }

    details = (
        _get(usage, "output_tokens_details")
        or _get(usage, "completion_tokens_details")
    )
    return {
        "prompt_tokens": int(
            _get(usage, "input_tokens")
            or _get(usage, "prompt_tokens")
            or 0
        ),
        "completion_tokens": int(
            _get(usage, "output_tokens")
            or _get(usage, "completion_tokens")
            or 0
        ),
        "reasoning_tokens": int(_get(details, "reasoning_tokens") or 0) if details else 0,
        "total_tokens": int(_get(usage, "total_tokens") or 0),
        "api_cost": _float_or_none(_get(usage, "cost")),
    }


def _get(value: Any, key: str):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
