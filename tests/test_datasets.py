import benchmark.datasets as datasets_mod


class FakeTemplate:
    def __init__(self, template):
        self.template = template

    def render(self, **row):
        return self.template.replace("{{ question }}", row["question"])


class FakeEnvironment:
    def __init__(self, **kwargs):
        pass

    def from_string(self, template):
        return FakeTemplate(template)


class FakeJinja2:
    StrictUndefined = object()
    Environment = FakeEnvironment


class FakeDatasets:
    def __init__(self):
        self.calls = []

    def load_dataset(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [{"question": "Q?", "answer": "A"}]


def test_dataset_expansion_does_not_trust_remote_code_by_default(monkeypatch):
    fake_datasets = FakeDatasets()
    monkeypatch.setattr(datasets_mod, "_ensure_jinja2", lambda: FakeJinja2)
    monkeypatch.setattr(datasets_mod, "_ensure_datasets", lambda: fake_datasets)

    expanded = datasets_mod.expand_dataset_task({
        "id": "ds",
        "category": "knowledge",
        "dataset": {"name": "example/ds"},
        "template": "{{ question }}",
        "scoring": {"type": "exact", "answer_field": "answer"},
    })

    assert expanded[0]["prompt"] == "Q?"
    assert fake_datasets.calls[0][1]["trust_remote_code"] is False


def test_dataset_expansion_allows_explicit_trust_remote_code(monkeypatch):
    fake_datasets = FakeDatasets()
    monkeypatch.setattr(datasets_mod, "_ensure_jinja2", lambda: FakeJinja2)
    monkeypatch.setattr(datasets_mod, "_ensure_datasets", lambda: fake_datasets)

    datasets_mod.expand_dataset_task({
        "id": "ds",
        "category": "knowledge",
        "dataset": {"name": "example/ds", "trust_remote_code": True},
        "template": "{{ question }}",
        "scoring": {"type": "exact", "answer_field": "answer"},
    })

    assert fake_datasets.calls[0][1]["trust_remote_code"] is True
