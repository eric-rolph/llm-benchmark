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
