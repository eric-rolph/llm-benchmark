"""
benchmark/loader.py — loads and validates task definitions from tasks/*.yaml
"""
from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).parent.parent / "tasks"

_REQUIRED_FIELDS = {"id", "prompt", "category", "scoring"}


def _validate_task(task: dict, source: str, index: int) -> None:
    """
    Raise ValueError if a task dict is missing required fields.
    Reports file name and task index so errors are immediately actionable.
    """
    missing = _REQUIRED_FIELDS - task.keys()
    if missing:
        raise ValueError(
            f"Task #{index} in '{source}' is missing required fields: {sorted(missing)}.\n"
            f"  Task data: {task}"
        )


def load_tasks(category_filter: str | None = None, validate: bool = True) -> list[dict]:
    if not TASKS_DIR.exists():
        raise FileNotFoundError(f"Tasks directory not found: {TASKS_DIR}")

    tasks: list[dict] = []
    for yaml_file in sorted(TASKS_DIR.glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        # Support both a bare YAML list and a dict with a 'tasks' key
        if isinstance(data, list):
            raw_tasks = data
            file_version = None
        elif isinstance(data, dict):
            raw_tasks = data.get("tasks", [])
            _meta = data.get("metadata", {})
            file_version = _meta.get("version") if isinstance(_meta, dict) else None
        else:
            continue  # skip empty / non-task YAML files
        for i, task in enumerate(raw_tasks):
            if validate:
                _validate_task(task, yaml_file.name, i)
            if category_filter is None or task.get("category") == category_filter:
                t = dict(task)
                if file_version is not None:
                    t["_version"] = file_version
                tasks.append(t)
    return tasks
