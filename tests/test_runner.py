from benchmark.runner import ModelRunner


class DummyBackend:
    name = "dummy"
    config = {}

    def get_openai_client(self):
        return object()


def test_hf_auto_config_is_opt_in(monkeypatch):
    calls = []

    def fake_fetch(self, model_id):
        calls.append(model_id)
        return {"temperature": 0.1}

    monkeypatch.setattr(ModelRunner, "_fetch_hf_config", fake_fetch)

    runner = ModelRunner(DummyBackend(), "org/model", {})

    assert calls == []
    assert runner.hf_generation_config == {}


def test_hf_auto_config_fetches_when_enabled(monkeypatch):
    calls = []

    def fake_fetch(self, model_id):
        calls.append(model_id)
        return {"temperature": 0.1}

    monkeypatch.setattr(ModelRunner, "_fetch_hf_config", fake_fetch)

    runner = ModelRunner(DummyBackend(), "org/model", {"hf_auto_config": True})

    assert calls == ["org/model"]
    assert runner.hf_generation_config == {"temperature": 0.1}


def test_run_task_aggregates_usage_metadata_across_repeated_runs(monkeypatch):
    runner = ModelRunner(DummyBackend(), "model-a", {"runs_per_task": 2})
    runs = [
        {
            "task_id": "task_a",
            "response": "A",
            "error": None,
            "ttft_ms": 10.0,
            "total_ms": 100.0,
            "tps": 5.0,
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "reasoning_tokens": 1,
            "total_tokens": 8,
            "api_cost": 0.001,
            "backend": "dummy",
        },
        {
            "task_id": "task_a",
            "response": "A",
            "error": None,
            "ttft_ms": 20.0,
            "total_ms": 200.0,
            "tps": 6.0,
            "prompt_tokens": 4,
            "completion_tokens": 6,
            "reasoning_tokens": 2,
            "total_tokens": 10,
            "api_cost": 0.002,
            "backend": "dummy",
        },
    ]

    monkeypatch.setattr(runner, "_run_once", lambda task: runs.pop(0))

    result = runner.run_task({"id": "task_a", "scoring": {"type": "exact", "value": "A"}})

    assert result["ttft_ms"] == 15.0
    assert result["prompt_tokens"] == 7
    assert result["completion_tokens"] == 11
    assert result["reasoning_tokens"] == 3
    assert result["total_tokens"] == 18
    assert result["api_cost"] == 0.003
    assert result["sample_count"] == 2
