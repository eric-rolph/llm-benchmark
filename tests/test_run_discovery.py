"""
tests/test_run_discovery.py — manual-model synthesis in discover_models.

A manual model id must yield exactly one (model, backend) pair: the
discovering backend's, or one synthesized phantom if nothing discovered it.
"""
from dataclasses import dataclass, field

from benchmark import session
from benchmark.backends.base import ModelInfo


@dataclass
class FakeBackend:
    name: str
    model_ids: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def is_available(self):
        return True

    def discover_models(self):
        return [
            ModelInfo(id=mid, name=mid, backend_name=self.name)
            for mid in self.model_ids
        ]


def _patch_backends(monkeypatch, backends_by_type):
    monkeypatch.setattr(
        session, "create_backend", lambda btype, cfg: backends_by_type[btype]
    )


def test_manual_model_discovered_by_one_backend_yields_single_pair(monkeypatch):
    _patch_backends(monkeypatch, {
        "a": FakeBackend("A", ["m1"]),
        "b": FakeBackend("B", []),
    })
    config = {
        "backends": {"a": {"enabled": True}, "b": {"enabled": True}},
        "models": ["m1"],
    }

    pairs = session.discover_models(config)

    assert [(m.id, b.name) for m, b in pairs] == [("m1", "A")]


def test_undiscovered_manual_model_synthesized_when_only_one_backend_is_reachable(monkeypatch):
    _patch_backends(monkeypatch, {"a": FakeBackend("A", [])})
    config = {
        "backends": {"a": {"enabled": True}},
        "models": ["m2"],
    }

    pairs = session.discover_models(config)

    assert len(pairs) == 1
    assert pairs[0][0].id == "m2"
    assert pairs[0][1].name == "A"


def test_undiscovered_manual_model_is_not_synthesized_across_multiple_backends(monkeypatch):
    _patch_backends(monkeypatch, {
        "a": FakeBackend("A", []),
        "b": FakeBackend("B", []),
    })
    config = {
        "backends": {"a": {"enabled": True}, "b": {"enabled": True}},
        "models": ["m2"],
    }

    pairs = session.discover_models(config)

    assert pairs == []


def test_manual_filter_still_drops_undeclared_models(monkeypatch):
    _patch_backends(monkeypatch, {"a": FakeBackend("A", ["m1", "other"])})
    config = {"backends": {"a": {"enabled": True}}, "models": ["m1"]}

    pairs = session.discover_models(config)

    assert [m.id for m, _ in pairs] == ["m1"]


def test_backend_qualified_manual_model_is_synthesized_on_requested_backend(monkeypatch):
    _patch_backends(monkeypatch, {
        "lm_studio": FakeBackend("LM Studio", []),
        "generic_openai": FakeBackend("Generic OpenAI", []),
    })
    config = {
        "backends": {
            "lm_studio": {"enabled": True},
            "generic_openai": {"enabled": True},
        },
        "models": [{"backend": "generic_openai", "id": "gpt-5.5"}],
    }

    pairs = session.discover_models(config)

    assert [(m.id, b.name) for m, b in pairs] == [("gpt-5.5", "Generic OpenAI")]


def test_backend_qualified_manual_model_does_not_match_other_backend(monkeypatch):
    _patch_backends(monkeypatch, {
        "lm_studio": FakeBackend("LM Studio", ["gpt-5.5"]),
        "generic_openai": FakeBackend("Generic OpenAI", []),
    })
    config = {
        "backends": {
            "lm_studio": {"enabled": True},
            "generic_openai": {"enabled": True},
        },
        "models": [{"backend": "generic_openai", "id": "gpt-5.5"}],
    }

    pairs = session.discover_models(config)

    assert [(m.id, b.name) for m, b in pairs] == [("gpt-5.5", "Generic OpenAI")]
