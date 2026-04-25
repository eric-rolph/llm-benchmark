"""
benchmark/loader.py — loads and validates task definitions from tasks/*.yaml

Supports two task types:
  1. Static tasks: standard {id, prompt, category, scoring} dicts
  2. Dataset tasks: {id, category, dataset, template, scoring} dicts that
     are expanded at load time via benchmark.datasets.expand_dataset_task()
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).parent.parent / "tasks"

_REQUIRED_FIELDS = {"id", "prompt", "category", "scoring"}

# Dataset tasks have a different required set — they use template instead of prompt
_DATASET_REQUIRED_FIELDS = {"id", "category", "dataset", "template", "scoring"}

logger = logging.getLogger(__name__)


def _validate_task(task: dict, source: str, index: int) -> None:
    """
    Raise ValueError if a task dict is missing required fields.
    Reports file name and task index so errors are immediately actionable.
    Dataset tasks (those with a 'dataset' key) are validated against a
    separate set of required fields.
    """
    if "dataset" in task:
        required = _DATASET_REQUIRED_FIELDS
    else:
        required = _REQUIRED_FIELDS
    missing = required - task.keys()
    if missing:
        raise ValueError(
            f"Task #{index} in '{source}' is missing required fields: {sorted(missing)}.\n"
            f"  Task data: {task}"
        )


def _expand_dataset_tasks(task: dict) -> list[dict]:
    """Expand a dataset-driven task into concrete task dicts, or return [task] for static tasks."""
    if "dataset" not in task:
        return [task]
    try:
        from benchmark.datasets import expand_dataset_task
        return expand_dataset_task(task)
    except ImportError as e:
        logger.warning(f"Skipping dataset task '{task.get('id', '?')}': {e}")
        return []
    except Exception as e:
        logger.warning(f"Error expanding dataset task '{task.get('id', '?')}': {e}")
        return []


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

            # Expand dataset tasks into concrete tasks
            expanded = _expand_dataset_tasks(task)

            for t in expanded:
                if category_filter is None or t.get("category") == category_filter:
                    t = dict(t)
                    if file_version is not None:
                        t["_version"] = file_version
                    tasks.append(t)
    return tasks
