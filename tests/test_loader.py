"""tests/test_loader.py — unit tests for benchmark/loader.py"""
import pytest

from benchmark.loader import load_tasks, _validate_task


# ── basic loading ──────────────────────────────────────────────────────────────

def test_load_tasks_returns_nonempty():
    tasks = load_tasks()
    assert len(tasks) > 0


def test_all_tasks_have_required_fields():
    tasks = load_tasks(validate=True)
    for t in tasks:
        assert "id" in t, f"Missing 'id' in task: {t}"
        assert "prompt" in t, f"Missing 'prompt' in task: {t}"
        assert "category" in t, f"Missing 'category' in task: {t}"
        assert "scoring" in t, f"Missing 'scoring' in task: {t}"


def test_tasks_carry_version():
    """Tasks from YAML files that declare metadata.version must have _version set."""
    tasks = load_tasks()
    versioned = [t for t in tasks if t.get("_version") is not None]
    assert len(versioned) > 0, "Expected at least one task with _version from metadata"


# ── category filter ────────────────────────────────────────────────────────────

def test_category_filter_limits_results():
    all_tasks = load_tasks()
    cats = {t["category"] for t in all_tasks}
    for cat in cats:
        filtered = load_tasks(category_filter=cat)
        assert len(filtered) > 0
        assert all(t["category"] == cat for t in filtered)


def test_category_filter_no_match_returns_empty():
    result = load_tasks(category_filter="nonexistent_category_xyz")
    assert result == []


def test_category_filter_subset_of_all():
    all_tasks = load_tasks()
    all_ids = {t["id"] for t in all_tasks}
    cats = {t["category"] for t in all_tasks}
    for cat in cats:
        filtered = load_tasks(category_filter=cat)
        assert {t["id"] for t in filtered}.issubset(all_ids)


# ── task IDs are unique ────────────────────────────────────────────────────────

def test_task_ids_are_unique():
    tasks = load_tasks()
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), f"Duplicate task IDs: {[x for x in ids if ids.count(x) > 1]}"


# ── validation ─────────────────────────────────────────────────────────────────

def test_validate_task_raises_on_missing_id():
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_task({"prompt": "x", "category": "y", "scoring": {}}, "test.yaml", 0)


def test_validate_task_raises_on_missing_prompt():
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_task({"id": "x", "category": "y", "scoring": {}}, "test.yaml", 0)


def test_validate_task_passes_with_all_required():
    # Should not raise
    _validate_task({"id": "x", "prompt": "q", "category": "math", "scoring": {"type": "numeric"}}, "test.yaml", 0)
