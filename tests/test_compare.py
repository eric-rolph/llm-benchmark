import json

import pytest

from benchmark.compare import compare_records, load_result_records


def _record(model: str, task: str, category: str, score: float, **extra):
    record = {
        "model_id": model,
        "task_id": task,
        "category": category,
        "score": score,
    }
    record.update(extra)
    return record


def test_compare_records_reports_model_and_task_deltas():
    baseline = [
        _record("model-a", "math_1", "math", 0.5, total_ms=100),
        _record("model-a", "knowledge_1", "knowledge", 1.0, total_ms=200),
        _record("model-b", "math_1", "math", 1.0),
    ]
    candidate = [
        _record("model-a", "math_1", "math", 1.0, total_ms=80),
        _record("model-a", "knowledge_1", "knowledge", 0.0, total_ms=250),
        _record("model-b", "math_1", "math", 1.0),
    ]

    summary = compare_records(baseline, candidate)

    assert summary["shared_count"] == 3
    assert summary["baseline_only_count"] == 0
    assert summary["candidate_only_count"] == 0

    model_a = next(row for row in summary["models"] if row["model_id"] == "model-a")
    assert model_a["baseline_score"] == pytest.approx(0.75)
    assert model_a["candidate_score"] == pytest.approx(0.5)
    assert model_a["score_delta"] == pytest.approx(-0.25)

    math_delta = next(row for row in summary["task_deltas"] if row["task_id"] == "math_1" and row["model_id"] == "model-a")
    assert math_delta["delta"] == pytest.approx(0.5)
    assert math_delta["total_ms_delta"] == pytest.approx(-20)


def test_compare_records_counts_unmatched_model_task_pairs():
    summary = compare_records(
        [_record("model-a", "task_1", "math", 1.0)],
        [_record("model-a", "task_2", "math", 1.0)],
    )

    assert summary["shared_count"] == 0
    assert summary["baseline_only_count"] == 1
    assert summary["candidate_only_count"] == 1


def test_load_result_records_supports_jsonl_and_saved_json(tmp_path):
    jsonl_path = tmp_path / "results.jsonl"
    jsonl_path.write_text(
        "\n".join([
            json.dumps(_record("model-a", "task_1", "math", 1.0)),
            json.dumps({"model": "model-b", "task_id": "task_2", "category": "coding", "score": 0.5}),
            "",
        ]),
        encoding="utf-8",
    )

    json_path = tmp_path / "results.json"
    json_path.write_text(
        json.dumps({
            "model-a": [
                {"task_id": "task_1", "category": "math", "score": 1.0},
            ],
        }),
        encoding="utf-8",
    )

    jsonl_records = load_result_records(jsonl_path)
    json_records = load_result_records(json_path)

    assert [(r["model_id"], r["task_id"], r["score"]) for r in jsonl_records] == [
        ("model-a", "task_1", 1.0),
        ("model-b", "task_2", 0.5),
    ]
    assert [(r["model_id"], r["task_id"], r["score"]) for r in json_records] == [
        ("model-a", "task_1", 1.0),
    ]


def test_load_result_records_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_result_records(tmp_path / "missing.jsonl")
