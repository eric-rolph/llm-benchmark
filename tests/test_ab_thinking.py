"""
tests/test_ab_thinking.py — harness-controlled thinking A/B (BACKLOG 4.2).

Covers the three pieces that make --ab-thinking work: the Ollama tri-state
think param, run_model's result_label arm separation, and the reporter's
arm pairing.
"""
from dataclasses import dataclass

from benchmark import session
from benchmark.backends.ollama import OllamaBackend
from benchmark.reporter import _pair_ab_results, print_ab_thinking_summary
from benchmark.result import cache_key


# ── Ollama think param tri-state ──────────────────────────────────────────────

def _ollama(config=None):
    return OllamaBackend({"name": "Ollama", **(config or {})})


def test_task_thinking_true_sends_think_true():
    params = _ollama().get_extra_chat_params({"thinking": True})
    assert params == {"extra_body": {"think": True}}


def test_task_thinking_false_sends_explicit_think_false():
    # The off-arm must say think=False on the wire — thinking models
    # default to thinking when the param is absent.
    params = _ollama({"thinking": True}).get_extra_chat_params({"thinking": False})
    assert params == {"extra_body": {"think": False}}


def test_config_thinking_used_when_task_silent():
    assert _ollama({"thinking": True}).get_extra_chat_params({}) == {
        "extra_body": {"think": True}
    }


def test_no_thinking_anywhere_sends_nothing():
    assert _ollama().get_extra_chat_params({}) == {}


def test_backend_declares_ab_support():
    assert OllamaBackend.supports_thinking_ab is True


# ── run_model result_label ────────────────────────────────────────────────────

@dataclass
class DummyModel:
    id: str = "model-a"


class DummyBackend:
    name = "dummy"
    config = {}


def _task(**updates):
    task = {
        "id": "task_a",
        "prompt": "Return A",
        "category": "knowledge",
        "scoring": {"type": "exact", "value": "A"},
    }
    task.update(updates)
    return task


def test_result_label_separates_arms(monkeypatch, tmp_path):
    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run_task(self, task):
            return {
                "task_id": task["id"],
                "response": "A",
                "error": None,
                "ttft_ms": None,
                "total_ms": 1.0,
                "tps": None,
                "completion_tokens": 1,
                "reasoning_tokens": 0,
                "backend": "dummy",
            }

    monkeypatch.setattr(session, "ModelRunner", FakeRunner)

    results = session.run_model(
        model_info=DummyModel(),
        backend=DummyBackend(),
        tasks=[_task(thinking=True)],
        bench_config={},
        cached_records={},
        jsonl_path=tmp_path / "results.jsonl",
        allow_code_exec=False,
        no_autoload=True,
        judge_client=None,
        judge_model=None,
        result_label="model-a [think]",
    )

    assert results[0]["model_id"] == "model-a [think]"


def test_arm_cache_keys_do_not_collide():
    on = cache_key("model-a [think]", _task(thinking=True))
    off = cache_key("model-a [no-think]", _task(thinking=False))
    assert on != off
    # the task fingerprint alone already differs — thinking is task content
    assert on[3] != off[3]


# ── reporter pairing ──────────────────────────────────────────────────────────

def _scored(score, reasoning_tokens=0, total_ms=100.0):
    return {
        "task": {"id": "t", "category": "knowledge"},
        "score": score,
        "passed": score >= 0.8,
        "reasoning_tokens": reasoning_tokens,
        "total_ms": total_ms,
        "tps": None,
    }


def test_pair_ab_results_groups_complete_pairs_only():
    all_results = {
        "qwen [think]": [_scored(1.0)],
        "qwen [no-think]": [_scored(0.5)],
        "solo-model": [_scored(0.7)],
        "orphan [think]": [_scored(0.9)],
    }

    pairs = _pair_ab_results(all_results)

    assert set(pairs) == {"qwen"}
    assert pairs["qwen"]["on"][0]["score"] == 1.0
    assert pairs["qwen"]["off"][0]["score"] == 0.5


def test_print_ab_thinking_summary_smoke():
    print_ab_thinking_summary({
        "qwen [think]": [_scored(1.0, reasoning_tokens=300, total_ms=900.0)],
        "qwen [no-think]": [_scored(0.5, total_ms=200.0)],
    })
    print_ab_thinking_summary({})  # no pairs — must be a no-op
