"""
benchmark/datasets.py — Hugging Face dataset integration with Jinja2 prompt templates.

Allows tasks to reference HF datasets instead of hardcoding prompts in YAML.
A single dataset task definition can dynamically expand into hundreds or
thousands of evaluation tasks at runtime.

YAML schema for dataset-driven tasks:

    - id: mmlu_anatomy
      category: knowledge
      dataset:
        name: "cais/mmlu"             # HF dataset ID
        subset: "anatomy"             # optional dataset config/subset
        split: "test"                 # default: "test"
        limit: 100                    # optional: cap number of examples
        shuffle: true                 # optional: randomize order
        seed: 42                      # optional: reproducible shuffle
      template: |
        Question: {{ question }}
        A) {{ choices[0] }}
        B) {{ choices[1] }}
        C) {{ choices[2] }}
        D) {{ choices[3] }}
        Answer with just the letter (A, B, C, or D).
      scoring:
        type: exact
        answer_field: "answer"        # column name in dataset → becomes the answer
        answer_map:                   # optional: map raw values to expected answers
          0: "A"
          1: "B"
          2: "C"
          3: "D"
"""
from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

# Lazy imports — these are optional deps, fail gracefully
_jinja2 = None
_hf_datasets = None


def _ensure_jinja2():
    global _jinja2
    if _jinja2 is None:
        try:
            import jinja2
            _jinja2 = jinja2
        except ImportError:
            raise ImportError(
                "Jinja2 is required for dataset-driven tasks. "
                "Install it with: pip install jinja2"
            )
    return _jinja2


def _ensure_datasets():
    global _hf_datasets
    if _hf_datasets is None:
        try:
            import datasets
            _hf_datasets = datasets
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for HF dataset tasks. "
                "Install it with: pip install datasets"
            )
    return _hf_datasets


def expand_dataset_task(task_def: dict) -> list[dict]:
    """
    Expand a single dataset-driven task definition into concrete task dicts.

    Each row from the HF dataset becomes one task with its prompt rendered
    from the Jinja2 template and its scoring answer extracted from the specified
    answer column.

    Returns a list of standard task dicts compatible with the rest of the
    benchmark pipeline.
    """
    ds_cfg = task_def.get("dataset", {})
    if not ds_cfg:
        return [task_def]  # Not a dataset task; return as-is

    jinja2 = _ensure_jinja2()
    datasets_lib = _ensure_datasets()

    # Load the dataset
    ds_name = ds_cfg["name"]
    subset = ds_cfg.get("subset")
    split = ds_cfg.get("split", "test")

    try:
        if subset:
            ds = datasets_lib.load_dataset(ds_name, subset, split=split, trust_remote_code=True)
        else:
            ds = datasets_lib.load_dataset(ds_name, split=split, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load dataset {ds_name}/{subset}: {e}")
        return []

    # Optional shuffle
    if ds_cfg.get("shuffle", False):
        seed = ds_cfg.get("seed", 42)
        ds = ds.shuffle(seed=seed)

    # Optional limit
    limit = ds_cfg.get("limit")
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    # Compile Jinja2 template
    template_str = task_def.get("template", "")
    if not template_str:
        raise ValueError(f"Dataset task '{task_def.get('id', '?')}' has no 'template' field")

    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_str)

    # Scoring config
    scoring = dict(task_def.get("scoring", {}))
    answer_field = scoring.pop("answer_field", None)
    answer_map = scoring.pop("answer_map", None)

    # System prompt (inherited from parent task def)
    system_prompt = task_def.get("system")

    base_id = task_def.get("id", "dataset")
    category = task_def.get("category", "knowledge")

    expanded: list[dict] = []
    for i, row in enumerate(ds):
        row_dict = dict(row)

        # Render prompt
        try:
            prompt = template.render(**row_dict)
        except Exception as e:
            logger.warning(f"Template render failed for {base_id}[{i}]: {e}")
            continue

        # Derive expected answer
        task_scoring = dict(scoring)
        if answer_field and answer_field in row_dict:
            raw_answer = row_dict[answer_field]
            if answer_map and raw_answer in answer_map:
                task_scoring["value"] = str(answer_map[raw_answer])
            else:
                task_scoring["value"] = str(raw_answer)

        # Generate a stable, unique task ID from base_id + row index
        row_hash = hashlib.md5(prompt.encode()).hexdigest()[:6]
        task_id = f"{base_id}_{i:04d}_{row_hash}"

        task = {
            "id": task_id,
            "category": category,
            "prompt": prompt,
            "scoring": task_scoring,
            "_dataset": ds_name,
            "_dataset_index": i,
        }
        if system_prompt:
            task["system"] = system_prompt
        if task_def.get("difficulty"):
            task["difficulty"] = task_def["difficulty"]

        expanded.append(task)

    logger.info(f"Expanded dataset task '{base_id}' → {len(expanded)} tasks from {ds_name}")
    return expanded
