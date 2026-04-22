"""benchmark/reporter.py - rich console output and result persistence."""
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


def _avg(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

CATEGORY_WEIGHTS: dict = {
    "coding":                1.5,
    "math":                  1.2,
    "reasoning":             1.2,
    "knowledge":             1.0,
    "instruction_following": 1.0,
    "summarization":         0.8,
    "writing":               0.8,
}


def _composite_score(results: list) -> float | None:
    """Weighted composite score across categories (harder categories carry more weight)."""
    if not results:
        return None
    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r["task"]["category"], []).append(r["score"])
    w_sum = 0.0
    w_tot = 0.0
    for cat, scores in by_cat.items():
        w = CATEGORY_WEIGHTS.get(cat, 1.0)
        w_sum += (sum(scores) / len(scores)) * w
        w_tot += w
    return w_sum / w_tot if w_tot > 0 else None

# ── per-task output ───────────────────────────────────────────────────────────

def print_task_result(result: dict):
    task = result["task"]
    score = result["score"]
    tps = result.get("tps")
    ttft = result.get("ttft_ms")
    error = result.get("error")

    if error:
        icon, style = "E", "red"
    elif score >= 1.0:
        icon, style = "✓", "green"
    elif score > 0:
        icon, style = "~", "yellow"
    else:
        icon, style = "✗", "red"

    tps_str = f"{tps:5.0f} t/s" if tps else "  N/A   "
    ttft_str = f"TTFT {ttft:.0f}ms" if ttft else ""

    console.print(
        f"  [{style}]{icon}[/{style}]  "
        f"[dim]{task['category']:22s}[/dim]"
        f"{task['id']:30s}  "
        f"score=[bold]{score:.1f}[/bold]  "
        f"{tps_str}  {ttft_str}"
    )
    if score < 1.0 and result.get("score_detail"):
        console.print(f"       [dim italic]{result['score_detail'][:110]}[/dim italic]")


# ── summary tables ────────────────────────────────────────────────────────────

def print_report(all_results: dict):
    console.print("\n")
    console.rule("[bold white]BENCHMARK SUMMARY[/bold white]")

    models = list(all_results.keys())
    categories = sorted({r["task"]["category"] for rs in all_results.values() for r in rs})

    # Accuracy table
    acc = Table(box=box.ROUNDED, title="Accuracy by Category", show_lines=True)
    acc.add_column("Category", style="bold", min_width=22)
    for m in models:
        acc.add_column(_short(m), justify="center", min_width=16)

    for cat in categories:
        row = [cat]
        for m in models:
            rs = [r for r in all_results[m] if r["task"]["category"] == cat]
            if rs:
                passed = sum(1 for r in rs if r["score"] >= 1.0)
                pct = sum(r["score"] for r in rs) / len(rs) * 100
                row.append(f"{passed}/{len(rs)}  ({pct:.0f}%)")
            else:
                row.append("—")
        acc.add_row(*row)

    total_row = ["[bold]TOTAL[/bold]"]
    for m in models:
        rs = all_results[m]
        passed = sum(1 for r in rs if r["score"] >= 1.0)
        pct = sum(r["score"] for r in rs) / len(rs) * 100 if rs else 0
        total_row.append(f"[bold]{passed}/{len(rs)}  ({pct:.0f}%)[/bold]")
    acc.add_row(*total_row)

    comp_row = ["[bold]Composite ★[/bold]"]
    for m in models:
        c = _composite_score(all_results[m])
        comp_row.append(f"[bold]{c * 100:.1f}%[/bold]" if c is not None else "—")
    acc.add_row(*comp_row)
    console.print(acc)

    # Performance table
    perf = Table(box=box.ROUNDED, title="Performance", show_lines=True)
    perf.add_column("Metric", style="bold", min_width=22)
    for m in models:
        perf.add_column(_short(m), justify="center", min_width=16)

    metrics = [
        ("Avg Tokens/sec",     lambda rs: _avg([r.get("tps") for r in rs])),
        ("Avg TTFT (ms)",      lambda rs: _avg([r.get("ttft_ms") for r in rs])),
        ("Avg Total (ms)",     lambda rs: _avg([r.get("total_ms") for r in rs])),
        ("Avg Think Tokens",   lambda rs: _avg([r.get("reasoning_tokens") for r in rs])),
    ]
    for label, fn in metrics:
        row = [label]
        for m in models:
            val = fn(all_results[m])
            row.append(f"{val:.1f}" if val is not None else "—")
        perf.add_row(*row)
    console.print(perf)

    # Latency histogram per category (first model only when single-model run, else each model)
    for m in models:
        lat = Table(box=box.ROUNDED, title=f"Latency by Category — {_short(m)}", show_lines=True)
        lat.add_column("Category",    style="bold", min_width=22)
        lat.add_column("Tasks",       justify="right", min_width=6)
        lat.add_column("Min (ms)",    justify="right", min_width=10)
        lat.add_column("Median (ms)", justify="right", min_width=12)
        lat.add_column("p95 (ms)",    justify="right", min_width=10)
        lat.add_column("Max (ms)",    justify="right", min_width=10)

        for cat in categories:
            vals = sorted([
                r["total_ms"] for r in all_results[m]
                if r["task"]["category"] == cat and r.get("total_ms") is not None
            ])
            if not vals:
                lat.add_row(cat, "—", "—", "—", "—", "—")
                continue
            p95_idx = int(len(vals) * 0.95)
            p95 = vals[min(p95_idx, len(vals) - 1)]
            med = statistics.median(vals)
            lat.add_row(cat, str(len(vals)),
                        f"{vals[0]:.0f}", f"{med:.0f}", f"{p95:.0f}", f"{vals[-1]:.0f}")
        console.print(lat)


def _short(model_id: str, max_len: int = 32) -> str:
    """Return the final path component of a model ID, truncated."""
    name = model_id.split("/")[-1]
    return name[:max_len] if len(name) > max_len else name


# ── incremental JSONL (crash-safe) ────────────────────────────────────────────

def append_jsonl(result: dict, path: Path) -> None:
    """
    Append one scored result as a single JSON line to `path`.
    Creates the file and parent directory on first write.
    Safe to call after every task so a mid-run crash loses no results.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model_id":  result.get("model_id", result.get("backend", "?")),
        "model":     result.get("backend", "?"),
        "task_id":   result["task"]["id"],
        "task_version": result["task"].get("_version"),
        "category":  result["task"]["category"],
        "score":     result["score"],
        "score_detail": result.get("score_detail", ""),
        "tps":       result.get("tps"),
        "ttft_ms":   result.get("ttft_ms"),
        "total_ms":  result.get("total_ms"),
        "response_preview": (result.get("response") or "")[:200],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def save_results(all_results: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON — full detail
    json_path = out / f"results_{ts}.json"
    payload: dict = {}
    for model, results in all_results.items():
        payload[model] = [
            {
                "task_id": r["task"]["id"],
                "category": r["task"]["category"],
                "score": r["score"],
                "score_detail": r.get("score_detail", ""),
                "tps": r.get("tps"),
                "ttft_ms": r.get("ttft_ms"),
                "total_ms": r.get("total_ms"),
                "completion_tokens": r.get("completion_tokens"),
                "reasoning_tokens": r.get("reasoning_tokens"),
                "response_preview": (r.get("response") or "")[:300],
            }
            for r in results
        ]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"\nJSON → [cyan]{json_path}[/cyan]")

    # CSV — flat for spreadsheet comparison
    csv_path = out / f"results_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "task_id", "category", "score",
                    "tps", "ttft_ms", "total_ms", "score_detail"])
        for model, results in all_results.items():
            for r in results:
                w.writerow([
                    model,
                    r["task"]["id"],
                    r["task"]["category"],
                    r["score"],
                    r.get("tps", ""),
                    r.get("ttft_ms", ""),
                    r.get("total_ms", ""),
                    r.get("score_detail", ""),
                ])
    console.print(f"CSV  → [cyan]{csv_path}[/cyan]")
