"""
tests/test_auditor.py — contamination audit probe (BACKLOG 4.3).
"""
from dataclasses import dataclass

from benchmark import auditor
from benchmark.auditor import (
    audit_contamination,
    expected_signals,
    print_audit_report,
    probe_task,
)


def _code_task(tid="code_x", test_code="assert lru_cache_get(c, 1) == 1\nprint('PASS')"):
    return {
        "id": tid,
        "prompt": "Implement a thread-safe LRU cache.",
        "category": "coding",
        "scoring": {"type": "code_exec", "test_code": test_code},
    }


# ── signal extraction ─────────────────────────────────────────────────────────

def test_expected_signals_extracts_solution_function_names():
    task = _code_task(test_code=(
        "c = lru_cache_new(2)\n"
        "lru_cache_put(c, 1, 'a')\n"
        "assert lru_cache_get(c, 1) == 'a'\n"
        "print(len(str('PASS')))"
    ))
    assert expected_signals(task) == ["lru_cache_new", "lru_cache_put", "lru_cache_get"]


def test_expected_signals_empty_without_test_code():
    assert expected_signals({"id": "t", "scoring": {"type": "contains"}}) == []


# ── the probe must never leak the problem statement ──────────────────────────

def test_probe_contains_task_id_but_not_problem_text():
    task = _code_task()
    probe = probe_task(task)
    assert task["id"] in probe["prompt"]
    assert "thread-safe" not in probe["prompt"].lower()
    assert "LRU" not in probe["prompt"]


# ── audit flow with a fake runner ─────────────────────────────────────────────

@dataclass
class DummyModel:
    id: str = "model-a"


class DummyBackend:
    name = "dummy"
    config = {}


def _patch_runner(monkeypatch, response_text):
    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run_task(self, task):
            return {"task_id": task["id"], "response": response_text, "error": None}

    monkeypatch.setattr(auditor, "ModelRunner", FakeRunner)


def test_contaminated_model_is_flagged(monkeypatch):
    _patch_runner(monkeypatch, (
        "def lru_cache_get(c, k): ...\n"
        "def lru_cache_put(c, k, v): ...\n"
        "def lru_cache_new(n): ..."
    ))
    task = _code_task(test_code=(
        "c = lru_cache_new(2)\nlru_cache_put(c, 1, 'a')\n"
        "assert lru_cache_get(c, 1) == 'a'\nprint('PASS')"
    ))

    report = audit_contamination([(DummyModel(), DummyBackend())], [task], {})

    row = report["model-a"][0]
    assert row["match_rate"] == 1.0
    assert row["flagged"] is True
    assert row["claimed_unknown"] is False


def test_honest_model_is_not_flagged(monkeypatch):
    _patch_runner(monkeypatch, "UNKNOWN")

    report = audit_contamination([(DummyModel(), DummyBackend())], [_code_task()], {})

    row = report["model-a"][0]
    assert row["match_rate"] == 0.0
    assert row["flagged"] is False
    assert row["claimed_unknown"] is True


def test_tasks_without_signals_are_skipped(monkeypatch):
    _patch_runner(monkeypatch, "anything")
    no_signal_task = {"id": "k1", "prompt": "Capital?", "category": "knowledge",
                      "scoring": {"type": "contains", "value": "Paris"}}

    report = audit_contamination([(DummyModel(), DummyBackend())], [no_signal_task], {})

    assert report == {}


def test_print_audit_report_smoke():
    print_audit_report({
        "model-a": [
            {"task_id": "code_x", "signals": ["f"], "matched": ["f"],
             "match_rate": 1.0, "claimed_unknown": False, "flagged": True,
             "error": None},
        ]
    })
    print_audit_report({})  # no auditable tasks — must not crash
