from dataclasses import dataclass

import pytest

import run


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


def test_resume_cache_key_changes_when_task_content_changes():
    original = _task()
    changed = _task(scoring={"type": "exact", "value": "B"})

    assert run._cache_key("model-a", original) != run._cache_key("model-a", changed)


def test_resume_cache_key_includes_task_version():
    assert run._cache_key("model-a", _task(_version=1)) != run._cache_key("model-a", _task(_version=2))


def test_run_model_hydrates_fully_cached_results_without_runner(monkeypatch, tmp_path):
    task = _task()
    key = run._cache_key("model-a", task)
    record = {
        "model_id": "model-a",
        "backend": "dummy",
        "task_id": task["id"],
        "task_version": task.get("_version"),
        "task_hash": key[3],
        "category": task["category"],
        "score": 1.0,
        "score_detail": "cached pass",
        "response_preview": "A",
    }

    def fail_model_runner(*args, **kwargs):
        raise AssertionError("cached resume should not instantiate ModelRunner")

    monkeypatch.setattr(run, "ModelRunner", fail_model_runner)

    results = run._run_model(
        model_info=DummyModel(),
        backend=DummyBackend(),
        tasks=[task],
        bench_config={},
        cached_records={key: record},
        jsonl_path=tmp_path / "results.jsonl",
        allow_code_exec=False,
        no_autoload=True,
        judge_client=None,
        judge_model=None,
    )

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["score_detail"] == "cached pass"


def test_pass_at_k_uses_samples_count_when_larger_than_k(monkeypatch, tmp_path):
    calls = []

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run_task_k(self, task, n):
            calls.append(n)
            return [
                {
                    "task_id": task["id"],
                    "response": "pass",
                    "error": None,
                    "ttft_ms": None,
                    "total_ms": 1.0,
                    "tps": None,
                    "completion_tokens": 1,
                    "reasoning_tokens": 0,
                    "backend": "dummy",
                },
                *[
                    {
                        "task_id": task["id"],
                        "response": "fail",
                        "error": None,
                        "ttft_ms": None,
                        "total_ms": 1.0,
                        "tps": None,
                        "completion_tokens": 1,
                        "reasoning_tokens": 0,
                        "backend": "dummy",
                    }
                    for _ in range(n - 1)
                ],
            ]

    monkeypatch.setattr(run, "ModelRunner", FakeRunner)
    task = _task(scoring={"type": "pass_at_k", "inner_type": "contains", "value": "pass", "k": 2, "n": 5})

    results = run._run_model(
        model_info=DummyModel(),
        backend=DummyBackend(),
        tasks=[task],
        bench_config={},
        cached_records={},
        jsonl_path=tmp_path / "results.jsonl",
        allow_code_exec=False,
        no_autoload=True,
        judge_client=None,
        judge_model=None,
    )

    assert calls == [5]
    assert results[0]["score_detail"].startswith("pass@2: 1/5")
