import json
from dataclasses import dataclass

import pytest

from benchmark import session
from benchmark.reporter import append_jsonl
from benchmark.result import cache_key, record_cache_key


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

    assert cache_key("model-a", original) != cache_key("model-a", changed)


def test_resume_cache_key_includes_task_version():
    assert cache_key("model-a", _task(_version=1)) != cache_key("model-a", _task(_version=2))


@pytest.mark.parametrize("updates", [{}, {"_version": 2}], ids=["versionless", "versioned"])
def test_resume_key_survives_jsonl_round_trip(tmp_path, updates):
    """Records written by the real append_jsonl must hydrate to the same key
    cache_key produces — a hand-built dict can't catch json null → "None"."""
    task = _task(**updates)
    result = {
        "task": task,
        "model_id": "model-a",
        "backend": "dummy",
        "score": 1.0,
        "pass_threshold": 0.99,
        "response": "A",
    }
    jsonl_path = tmp_path / "results.jsonl"
    append_jsonl(result, jsonl_path)

    record = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert record_cache_key(record) == cache_key("model-a", task)


def test_run_model_hydrates_fully_cached_results_without_runner(monkeypatch, tmp_path):
    task = _task()
    key = cache_key("model-a", task)
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

    monkeypatch.setattr(session, "ModelRunner", fail_model_runner)

    results = session.run_model(
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


def test_agent_loop_without_allow_code_exec_does_not_instantiate_runner(monkeypatch, tmp_path):
    task = _task(
        category="agent_loop",
        execution_surface="observed_agent_loop",
        scoring={"type": "agent_loop"},
    )

    def fail_model_runner(*args, **kwargs):
        raise AssertionError("disabled agent_loop should not instantiate ModelRunner")

    monkeypatch.setattr(session, "ModelRunner", fail_model_runner)

    results = session.run_model(
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

    assert results[0]["score"] == 0.0
    assert "disabled" in results[0]["score_detail"]


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

    monkeypatch.setattr(session, "ModelRunner", FakeRunner)
    task = _task(scoring={"type": "pass_at_k", "inner_type": "contains", "value": "pass", "k": 2, "n": 5})

    results = session.run_model(
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


def test_api_cost_budget_accumulates_only_reported_costs():
    budget = session.ApiCostBudget(limit=0.003)

    budget.add_result({"api_cost": 0.001})
    budget.add_result({"api_cost": None})
    budget.add_result({"api_cost": 0.002})

    assert budget.spent == pytest.approx(0.003)
    assert budget.exhausted


def test_resolve_api_cost_budget_prefers_cli_override():
    budget = session.resolve_api_cost_budget({"max_api_cost": 3.5}, cli_limit=1.25)

    assert budget is not None
    assert budget.limit == pytest.approx(1.25)


def test_run_model_stops_before_next_uncached_task_when_api_cost_budget_is_exhausted(monkeypatch, tmp_path):
    calls = []

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run_task(self, task):
            calls.append(task["id"])
            return {
                "task_id": task["id"],
                "response": "A",
                "error": None,
                "ttft_ms": None,
                "total_ms": 1.0,
                "tps": None,
                "completion_tokens": 1,
                "reasoning_tokens": 0,
                "api_cost": 0.0025,
                "backend": "dummy",
            }

    monkeypatch.setattr(session, "ModelRunner", FakeRunner)
    budget = session.ApiCostBudget(limit=0.002)
    tasks = [_task(id="task_a"), _task(id="task_b")]

    results = session.run_model(
        model_info=DummyModel(),
        backend=DummyBackend(),
        tasks=tasks,
        bench_config={},
        cached_records={},
        jsonl_path=tmp_path / "results.jsonl",
        allow_code_exec=False,
        no_autoload=True,
        judge_client=None,
        judge_model=None,
        api_cost_budget=budget,
    )

    assert calls == ["task_a"]
    assert [r["task"]["id"] for r in results] == ["task_a"]
    assert budget.spent == pytest.approx(0.0025)
    assert budget.exhausted
