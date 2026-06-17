"""Shared evaluation policy helpers."""
from __future__ import annotations

DEFAULT_PASS_THRESHOLD = 0.8
NON_LEADERBOARD_TIERS = {"smoke", "diagnostic"}

# Composite-score weights: harder categories carry more weight.
CATEGORY_WEIGHTS: dict = {
    "agent_loop":            1.8,
    "coding":                1.5,
    "repo_patch":            1.6,
    "math":                  1.2,
    "reasoning":             1.2,
    "agentic":               1.4,
    "knowledge":             1.0,
    "instruction_following": 1.0,
    "summarization":         0.8,
    "writing":               0.8,
}

# Expected reasoning-token budget per category (E3-Score baseline).
# Derived from EffiReason-Bench: a "right-sized" model should need roughly
# this many reasoning tokens to solve tasks in each category.
# Non-thinking models (reasoning_tokens == 0) are excluded from E3 computation.
E3_EXPECTED_TOKENS: dict = {
    "agent_loop":            5000,
    "coding":                2000,
    "repo_patch":            2500,
    "agentic":               1500,
    "math":                  600,
    "reasoning":             600,
    "knowledge":             200,
    "instruction_following": 200,
    "summarization":         400,
    "writing":               400,
}


def task_tier(task: dict | None) -> str:
    """Return a normalized benchmark tier for a task or persisted record."""
    task = task or {}
    raw = task.get("benchmark_tier", task.get("tier", "leaderboard"))
    value = str(raw or "leaderboard").strip().lower()
    return value or "leaderboard"


def is_leaderboard_task(task: dict | None) -> bool:
    """True for tasks that should contribute to headline/core comparisons."""
    task = task or {}
    if task_tier(task) in NON_LEADERBOARD_TIERS:
        return False
    contamination = str(task.get("contamination_risk", "") or "").strip().lower()
    return contamination != "high"


def leaderboard_results(results: list[dict]) -> list[dict]:
    """Filter scored results or flat records down to headline-eligible rows."""
    filtered = []
    for result in results:
        task = result.get("task", result)
        if is_leaderboard_task(task):
            filtered.append(result)
    return filtered


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
