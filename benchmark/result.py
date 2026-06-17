"""
benchmark/result.py — the single definition of the persisted result record.

A *result* is the in-memory dict produced by ModelRunner and the scorer,
carrying the full task under "task". A *record* is its flat, JSON-safe
persisted form — one JSONL line, one entry in results_*.json. Both
directions of the mapping live here, together with the resume cache keys,
so a new metric is added in exactly one place (plus the runner that
measures it).
"""
from __future__ import annotations

from benchmark.evaluation import annotate_pass, result_passed, task_pass_threshold, task_tier
from benchmark.utils import task_fingerprint

RESPONSE_PREVIEW_CHARS = 4000


def to_record(result: dict) -> dict:
    """Flatten a scored result into its persisted record form."""
    task = result["task"]
    return {
        "model_id":  result.get("model_id", "?"),
        "model":     result.get("model_id", "?"),  # legacy readers
        "backend":   result.get("backend", "?"),
        "task_id":   task["id"],
        "task_version": task.get("_version"),
        "task_hash": task_fingerprint(task),
        "category":  task["category"],
        "benchmark_tier": task_tier(task),
        "contamination_risk": task.get("contamination_risk"),
        "execution_surface": task.get("execution_surface"),
        "source_signal": task.get("source_signal", task.get("_signal_source")),
        "human_minutes_estimate": task.get("human_minutes_estimate"),
        "criticisms_addressed": task.get("criticisms_addressed"),
        "signal_snapshot": task.get("_signal_snapshot"),
        "release": task.get("_release"),
        "scoring_type": task.get("scoring", {}).get("type"),
        "score":     result["score"],
        "pass_threshold": result.get("pass_threshold"),
        "passed": result_passed(result),
        "score_detail": result.get("score_detail", ""),
        "tps":       result.get("tps"),
        "ttft_ms":   result.get("ttft_ms"),
        "total_ms":  result.get("total_ms"),
        "completion_tokens": result.get("completion_tokens"),
        "reasoning_tokens":  result.get("reasoning_tokens"),
        "peak_vram_mb":      result.get("peak_vram_mb"),
        "avg_gpu_util":      result.get("avg_gpu_util"),
        "logprob_detail":    result.get("logprob_detail"),
        "hf_generation_config": result.get("hf_generation_config"),
        "execution_trace": result.get("execution_trace"),
        "response_preview": (result.get("response") or "")[:RESPONSE_PREVIEW_CHARS],
    }


def from_record(task: dict, record: dict) -> dict:
    """Rebuild a scored result from a persisted record (resume hydration)."""
    result = {
        "task_id": task["id"],
        "task": task,
        "response": record.get("response_preview", ""),
        "error": None,
        "score": float(record.get("score", 0.0)),
        "max_score": 1.0,
        "pass_threshold": record.get("pass_threshold", task_pass_threshold(task)),
        "passed": record.get("passed"),
        "score_detail": record.get("score_detail", ""),
        "tps": record.get("tps"),
        "ttft_ms": record.get("ttft_ms"),
        "total_ms": record.get("total_ms"),
        "completion_tokens": record.get("completion_tokens"),
        "reasoning_tokens": record.get("reasoning_tokens", 0),
        "peak_vram_mb": record.get("peak_vram_mb"),
        "avg_gpu_util": record.get("avg_gpu_util"),
        "backend": record.get("backend", "?"),
        "model_id": record.get("model_id", record.get("model", "?")),
        "logprob_detail": record.get("logprob_detail"),
        "hf_generation_config": record.get("hf_generation_config"),
        "execution_trace": record.get("execution_trace"),
    }
    if result["passed"] is None:
        annotate_pass(result)
    return result


def cache_key(model_id: str, task: dict) -> tuple[str, str, str, str]:
    """Key cached results by model, task id, declared version, and task content."""
    return (
        model_id,
        task["id"],
        str(task.get("_version") or ""),
        task_fingerprint(task),
    )


def record_cache_key(record: dict) -> tuple[str, str, str, str]:
    """Build the resume key for a persisted record."""
    # JSON round-trips a missing version as an explicit null — normalize to ""
    # so it matches cache_key for versionless tasks.
    return (
        record.get("model_id", record.get("model", "")),
        record.get("task_id", ""),
        str(record.get("task_version") or ""),
        record.get("task_hash", ""),
    )
