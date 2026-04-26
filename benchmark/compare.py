"""Compare two benchmark result files."""
from __future__ import annotations

import json
from pathlib import Path

from rich import box
from rich.table import Table

from benchmark.console import make_console
from benchmark.reporter import CATEGORY_WEIGHTS
from benchmark.utils import _avg

console = make_console()


def _normalise_record(record: dict, model_hint: str | None = None) -> dict:
    model = record.get("model_id") or record.get("model") or model_hint or "?"
    return {
        "model_id": model,
        "task_id": record["task_id"],
        "task_version": record.get("task_version"),
        "task_hash": record.get("task_hash"),
        "category": record.get("category", "?"),
        "score": float(record.get("score", 0.0)),
        "tps": record.get("tps"),
        "ttft_ms": record.get("ttft_ms"),
        "total_ms": record.get("total_ms"),
        "score_detail": record.get("score_detail", ""),
    }


def load_result_records(path: str | Path) -> list[dict]:
    """Load flat records from JSONL or saved JSON result files."""
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    if result_path.suffix.lower() == ".jsonl":
        records = []
        with result_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(_normalise_record(json.loads(line)))
        return records

    data = json.loads(result_path.read_text(encoding="utf-8"))
    records = []
    if isinstance(data, dict):
        for model, model_records in data.items():
            if isinstance(model_records, list):
                records.extend(_normalise_record(record, model) for record in model_records)
    elif isinstance(data, list):
        records = [_normalise_record(record) for record in data]
    return records


def _by_model_task(records: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (record["model_id"], record["task_id"]): record
        for record in records
    }


def _mean_score(records: list[dict]) -> float | None:
    return _avg([record.get("score") for record in records])


def _composite(records: list[dict]) -> float | None:
    by_cat: dict[str, list[float]] = {}
    for record in records:
        by_cat.setdefault(record["category"], []).append(record["score"])
    if not by_cat:
        return None
    weighted = 0.0
    total_weight = 0.0
    for category, scores in by_cat.items():
        weight = CATEGORY_WEIGHTS.get(category.lower().strip(), 1.0)
        weighted += (sum(scores) / len(scores)) * weight
        total_weight += weight
    return weighted / total_weight if total_weight else None


def compare_records(baseline: list[dict], candidate: list[dict]) -> dict:
    """Return aggregate and per-task deltas between two result sets."""
    baseline_map = _by_model_task(baseline)
    candidate_map = _by_model_task(candidate)
    shared_keys = sorted(set(baseline_map) & set(candidate_map))

    task_deltas = []
    for key in shared_keys:
        old = baseline_map[key]
        new = candidate_map[key]
        task_deltas.append({
            "model_id": key[0],
            "task_id": key[1],
            "category": new.get("category") or old.get("category"),
            "baseline_score": old["score"],
            "candidate_score": new["score"],
            "delta": new["score"] - old["score"],
            "baseline_total_ms": old.get("total_ms"),
            "candidate_total_ms": new.get("total_ms"),
            "total_ms_delta": (
                new["total_ms"] - old["total_ms"]
                if new.get("total_ms") is not None and old.get("total_ms") is not None
                else None
            ),
        })

    models = sorted({record["model_id"] for record in baseline + candidate})
    model_rows = []
    for model in models:
        old_records = [r for r in baseline if r["model_id"] == model]
        new_records = [r for r in candidate if r["model_id"] == model]
        old_score = _mean_score(old_records)
        new_score = _mean_score(new_records)
        old_comp = _composite(old_records)
        new_comp = _composite(new_records)
        model_rows.append({
            "model_id": model,
            "baseline_tasks": len(old_records),
            "candidate_tasks": len(new_records),
            "baseline_score": old_score,
            "candidate_score": new_score,
            "score_delta": (
                new_score - old_score
                if new_score is not None and old_score is not None
                else None
            ),
            "baseline_composite": old_comp,
            "candidate_composite": new_comp,
            "composite_delta": (
                new_comp - old_comp
                if new_comp is not None and old_comp is not None
                else None
            ),
        })

    return {
        "baseline_count": len(baseline),
        "candidate_count": len(candidate),
        "shared_count": len(shared_keys),
        "baseline_only_count": len(set(baseline_map) - set(candidate_map)),
        "candidate_only_count": len(set(candidate_map) - set(baseline_map)),
        "models": model_rows,
        "task_deltas": task_deltas,
    }


def _fmt_pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.1f}%"


def print_comparison(summary: dict, top_n: int = 10) -> None:
    """Render a comparison summary to the console."""
    console.rule("[bold white]RESULT COMPARISON[/bold white]")
    console.print(
        f"[dim]Shared tasks: {summary['shared_count']} | "
        f"baseline-only: {summary['baseline_only_count']} | "
        f"candidate-only: {summary['candidate_only_count']}[/dim]\n"
    )

    model_table = Table(box=box.ROUNDED, title="Model Summary", show_lines=True)
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Base", justify="right")
    model_table.add_column("New", justify="right")
    model_table.add_column("Delta", justify="right")
    model_table.add_column("Composite Delta", justify="right")
    model_table.add_column("Tasks", justify="right")
    for row in summary["models"]:
        model_table.add_row(
            row["model_id"],
            _fmt_pct(row["baseline_score"]),
            _fmt_pct(row["candidate_score"]),
            _fmt_pct(row["score_delta"], signed=True),
            _fmt_pct(row["composite_delta"], signed=True),
            f"{row['baseline_tasks']} -> {row['candidate_tasks']}",
        )
    console.print(model_table)

    swings = sorted(summary["task_deltas"], key=lambda row: abs(row["delta"]), reverse=True)
    if not swings:
        return

    task_table = Table(box=box.ROUNDED, title=f"Largest Task Score Changes (top {top_n})", show_lines=False)
    task_table.add_column("Model", style="cyan")
    task_table.add_column("Category", style="dim")
    task_table.add_column("Task")
    task_table.add_column("Base", justify="right")
    task_table.add_column("New", justify="right")
    task_table.add_column("Delta", justify="right")
    for row in swings[:top_n]:
        task_table.add_row(
            row["model_id"],
            row["category"],
            row["task_id"],
            _fmt_pct(row["baseline_score"]),
            _fmt_pct(row["candidate_score"]),
            _fmt_pct(row["delta"], signed=True),
        )
    console.print(task_table)


def compare_result_files(baseline_path: str | Path, candidate_path: str | Path, top_n: int = 10) -> dict:
    summary = compare_records(
        load_result_records(baseline_path),
        load_result_records(candidate_path),
    )
    print_comparison(summary, top_n=top_n)
    return summary
