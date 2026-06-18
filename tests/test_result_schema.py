"""
tests/test_result_schema.py — benchmark/result.py is the single source of
the persisted record shape; results must survive the round trip.
"""
import json

from benchmark.result import RESPONSE_PREVIEW_CHARS, from_record, to_record


def _result(**updates):
    result = {
        "task": {
            "id": "t1",
            "prompt": "Return A",
            "category": "knowledge",
            "benchmark_tier": "smoke",
            "contamination_risk": "high",
            "source_signal": "swe-terminal-criticism",
            "human_minutes_estimate": 15,
            "criticisms_addressed": ["hidden_tests", "hermetic_workspace"],
            "scoring": {"type": "exact", "value": "A"},
            "_version": 3,
        },
        "task_id": "t1",
        "model_id": "model-a",
        "backend": "dummy",
        "run_fingerprint": "abc123",
        "response": "A",
        "error": None,
        "score": 1.0,
        "score_std": 0.1,
        "pass_threshold": 0.99,
        "passed": True,
        "score_detail": "Exact match",
        "tps": 42.5,
        "ttft_ms": 12.0,
        "total_ms": 100.0,
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "reasoning_tokens": 7,
        "total_tokens": 12,
        "api_cost": 0.0042,
        "sample_count": 3,
        "peak_vram_mb": 2048,
        "avg_gpu_util": 55.0,
        "logprob_detail": None,
        "hf_generation_config": {},
        "execution_trace": {"events": []},
    }
    result.update(updates)
    return result


def test_record_round_trip_preserves_scoring_and_metrics():
    result = _result()
    record = json.loads(json.dumps(to_record(result)))  # through real JSON
    hydrated = from_record(result["task"], record)

    for field in (
        "model_id", "backend", "score", "passed", "score_detail",
        "run_fingerprint", "score_std", "tps", "ttft_ms", "total_ms", "prompt_tokens", "completion_tokens",
        "reasoning_tokens", "total_tokens", "api_cost", "sample_count", "peak_vram_mb", "avg_gpu_util",
    ):
        assert hydrated[field] == result[field], field
    assert hydrated["response"] == "A"
    assert hydrated["task"] is result["task"]


def test_record_carries_task_identity_for_resume():
    record = to_record(_result())
    assert record["task_id"] == "t1"
    assert record["task_version"] == 3
    assert record["task_hash"]
    assert record["category"] == "knowledge"
    assert record["benchmark_tier"] == "smoke"
    assert record["contamination_risk"] == "high"
    assert record["source_signal"] == "swe-terminal-criticism"
    assert record["human_minutes_estimate"] == 15
    assert record["criticisms_addressed"] == ["hidden_tests", "hermetic_workspace"]
    assert record["scoring_type"] == "exact"


def test_passed_is_recomputed_when_absent_from_record():
    record = to_record(_result())
    del record["passed"]
    hydrated = from_record(_result()["task"], record)
    assert hydrated["passed"] is True  # re-derived from score vs threshold


def test_response_preview_is_long_enough_to_debug_patch_failures():
    response = "x" * (RESPONSE_PREVIEW_CHARS + 100)
    record = to_record(_result(response=response))

    assert len(record["response_preview"]) == RESPONSE_PREVIEW_CHARS
    assert RESPONSE_PREVIEW_CHARS >= 4000


def test_run_py_shim_still_exposes_main():
    import run

    assert callable(run.main)
