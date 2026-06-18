import pytest

from benchmark.cli import _filter_tasks_by_ids


def test_filter_tasks_by_ids_accepts_multiple_task_ids():
    tasks = [
        {"id": "a", "category": "coding"},
        {"id": "b", "category": "coding"},
        {"id": "c", "category": "coding"},
    ]

    assert _filter_tasks_by_ids(tasks, ["a", "c"]) == [tasks[0], tasks[2]]


def test_filter_tasks_by_ids_reports_missing_ids():
    tasks = [{"id": "a", "category": "coding"}]

    with pytest.raises(ValueError, match="Task ID\\(s\\) not found: b, c"):
        _filter_tasks_by_ids(tasks, ["b", "c"])
