"""Shared evaluation policy helpers."""
from __future__ import annotations

DEFAULT_PASS_THRESHOLD = 0.8


def task_pass_threshold(task: dict | None) -> float:
    """Return the pass threshold for a task, clamped to [0.0, 1.0]."""
    task = task or {}
    scoring = task.get("scoring", {})
    if not isinstance(scoring, dict):
        scoring = {}

    raw = task.get("pass_threshold", scoring.get("pass_threshold", DEFAULT_PASS_THRESHOLD))
    try:
        threshold = float(raw)
    except (TypeError, ValueError):
        threshold = DEFAULT_PASS_THRESHOLD
    return min(max(threshold, 0.0), 1.0)


def annotate_pass(result: dict) -> dict:
    """Attach pass threshold and boolean pass status to a scored result."""
    threshold = task_pass_threshold(result.get("task"))
    try:
        score = float(result.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    result["pass_threshold"] = threshold
    result["passed"] = score >= threshold
    return result


def result_passed(result: dict) -> bool:
    """Return pass/fail for a result, computing it if older records lack it."""
    if "passed" in result:
        return bool(result["passed"])
    try:
        score = float(result.get("score", 0.0) or 0.0)
        threshold = float(result.get("pass_threshold", DEFAULT_PASS_THRESHOLD))
    except (TypeError, ValueError):
        return False
    return score >= threshold
