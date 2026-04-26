"""
benchmark/utils.py — shared utilities used across runner, scorer, and backends.

Centralised here to avoid the _THINK_RE / strip_thinking duplication that
previously existed in both runner.py and scorer.py.
"""
from __future__ import annotations

import re
import hashlib
import json

# Matches <think>…</think> or <thinking>…</thinking> blocks (case-insensitive).
# Used by both the runner (to separate reasoning from answer)
# and the scorer (belt-and-suspenders strip before scoring).
_THINK_RE = re.compile(
    r"<think>.*?</think>|<thinking>.*?</thinking>",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> / <thinking>…</thinking> blocks and strip whitespace."""
    return _THINK_RE.sub("", text).strip()


def _avg(values: list) -> float | None:
    """Return the mean of non-None values, or None if no valid values."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def task_fingerprint(task: dict) -> str:
    """Return a stable hash for task content that affects scoring/results."""
    public_task = {
        key: value
        for key, value in task.items()
        if not key.startswith("_")
    }
    payload = json.dumps(public_task, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
