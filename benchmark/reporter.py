"""benchmark/reporter.py - rich console output and result persistence."""
import csv
import json
import math
import statistics
from datetime import datetime
from html import escape
from pathlib import Path

from rich import box
from rich.table import Table

from benchmark.console import make_console
from benchmark.evaluation import CATEGORY_WEIGHTS, E3_EXPECTED_TOKENS, leaderboard_results, result_passed
from benchmark.result import to_record
from benchmark.utils import _avg

console = make_console()


def _csv_value(value):
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def _e3_score(score: float, reasoning_tokens: int | None, category: str) -> float | None:
    """
    E3-Score (EffiReason-Bench variant):
        E3 = accuracy × log(expected + 1) / log(actual + 1)
    Rewards correct answers achieved with fewer reasoning tokens.
    Returns None when reasoning_tokens == 0 (non-thinking model) so we
    don't penalise models that produce no chain-of-thought.
    """
    if reasoning_tokens is None or reasoning_tokens == 0:
        return None
    expected = E3_EXPECTED_TOKENS.get(category.lower().strip(), 500)
    actual   = max(reasoning_tokens, 1)
    return score * math.log(expected + 1) / math.log(actual + 1)


def _sum_or_none(values: list) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) if present else None


def _coverage_counts(results: list, expected_tasks: list[dict] | None = None) -> tuple[int, int]:
    completed_ids = {r["task"]["id"] for r in results}
    if expected_tasks is None:
        return len(completed_ids), len(completed_ids)
    expected_ids = {task["id"] for task in expected_tasks}
    return len(completed_ids & expected_ids), len(expected_ids)


def _composite_score(
    results: list,
    core_only: bool = True,
    expected_tasks: list[dict] | None = None,
) -> float | None:
    """Weighted composite score across categories (harder categories carry more weight)."""
    completed, expected = _coverage_counts(results, expected_tasks)
    if expected_tasks is not None and completed < expected:
        return None
    if core_only:
        results = leaderboard_results(results)
    if not results:
        return None
    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r["task"]["category"], []).append(r["score"])
    w_sum = 0.0
    w_tot = 0.0
    for cat, scores in by_cat.items():
        w = CATEGORY_WEIGHTS.get(cat.lower().strip(), 1.0)
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
    elif result_passed(result):
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
    progress_detail = _agent_loop_progress_detail(result)
    if score < 1.0 and progress_detail:
        console.print(f"       [dim italic]{progress_detail}[/dim italic]")


def _agent_loop_progress_detail(result: dict) -> str:
    passed = result.get("agent_loop_progress_passed")
    total = result.get("agent_loop_progress_total")
    if passed is None or total is None:
        return ""
    detail = f"progress={passed}/{total}"
    termination = result.get("agent_loop_termination")
    if termination:
        detail += f"; termination={termination}"
    return detail


# ── summary tables ────────────────────────────────────────────────────────────

def print_report(all_results: dict, expected_tasks: list[dict] | None = None):
    console.print("\n")
    console.rule("[bold white]BENCHMARK SUMMARY[/bold white]")

    models = list(all_results.keys())
    categories = sorted(
        {r["task"]["category"] for rs in all_results.values() for r in rs}
        | {t["category"] for t in (expected_tasks or [])}
    )

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
                passed = sum(1 for r in rs if result_passed(r))
                pct = sum(r["score"] for r in rs) / len(rs) * 100
                stds = [r["score_std"] for r in rs if r.get("score_std") is not None]
                if stds:
                    avg_std = sum(stds) / len(stds) * 100
                    row.append(f"{passed}/{len(rs)}  ({pct:.0f}% ±{avg_std:.0f}%)")
                else:
                    row.append(f"{passed}/{len(rs)}  ({pct:.0f}%)")
            else:
                row.append("—")
        acc.add_row(*row)

    total_row = ["[bold]TOTAL[/bold]"]
    for m in models:
        rs = all_results[m]
        passed = sum(1 for r in rs if result_passed(r))
        pct = sum(r["score"] for r in rs) / len(rs) * 100 if rs else 0
        total_row.append(f"[bold]{passed}/{len(rs)}  ({pct:.0f}%)[/bold]")
    acc.add_row(*total_row)

    coverage_row = ["[dim]Coverage[/dim]"]
    for m in models:
        completed, expected = _coverage_counts(all_results[m], expected_tasks)
        if expected and completed < expected:
            coverage_row.append(f"[yellow]{completed}/{expected} tasks[/yellow]")
        elif expected:
            coverage_row.append(f"[dim]{completed}/{expected} tasks[/dim]")
        else:
            coverage_row.append("[dim]—[/dim]")
    acc.add_row(*coverage_row)

    # Headline score row: exclude smoke/diagnostic tasks and high-contamination tasks.
    clean_row = ["[dim]Leaderboard core[/dim]"]
    for m in models:
        rs = leaderboard_results(all_results[m])
        if rs:
            passed = sum(1 for r in rs if result_passed(r))
            pct = sum(r["score"] for r in rs) / len(rs) * 100
            clean_row.append(f"[dim]{passed}/{len(rs)}  ({pct:.0f}%)[/dim]")
        else:
            clean_row.append("[dim]—[/dim]")
    acc.add_row(*clean_row)

    comp_row = ["[bold]Composite ★[/bold]"]
    for m in models:
        c = _composite_score(all_results[m], expected_tasks=expected_tasks)
        completed, expected = _coverage_counts(all_results[m], expected_tasks)
        if c is not None:
            comp_row.append(f"[bold]{c * 100:.1f}%[/bold]")
        elif expected and completed < expected:
            comp_row.append(f"[yellow]incomplete {completed}/{expected}[/yellow]")
        else:
            comp_row.append("—")
    acc.add_row(*comp_row)
    console.print(acc)

    surfaces = sorted({
        r["task"].get("execution_surface")
        for rs in all_results.values() for r in rs
        if r["task"].get("execution_surface")
    })
    if surfaces:
        surf = Table(box=box.ROUNDED, title="Accuracy by Execution Surface", show_lines=True)
        surf.add_column("Execution Surface", style="bold", min_width=22)
        for m in models:
            surf.add_column(_short(m), justify="center", min_width=16)

        for surface in surfaces:
            row = [surface]
            for m in models:
                rs = [r for r in all_results[m] if r["task"].get("execution_surface") == surface]
                if rs:
                    passed = sum(1 for r in rs if result_passed(r))
                    pct = sum(r["score"] for r in rs) / len(rs) * 100
                    row.append(f"{passed}/{len(rs)}  ({pct:.0f}%)")
                else:
                    row.append("—")
            surf.add_row(*row)
        console.print(surf)

    # Performance table
    perf = Table(box=box.ROUNDED, title="Performance", show_lines=True)
    perf.add_column("Metric", style="bold", min_width=22)
    for m in models:
        perf.add_column(_short(m), justify="center", min_width=16)

    metrics = [
        ("Avg Tokens/sec",     lambda rs: _avg([r.get("tps") for r in rs])),
        ("Avg TTFT (ms)",      lambda rs: _avg([r.get("ttft_ms") for r in rs])),
        ("Avg Total (ms)",     lambda rs: _avg([r.get("total_ms") for r in rs])),
        ("Avg Output Tokens",  lambda rs: _avg([r.get("completion_tokens") for r in rs])),
        ("Avg Think Tokens",   lambda rs: _avg([r.get("reasoning_tokens") for r in rs])),
        ("Total API Cost",     lambda rs: _sum_or_none([r.get("api_cost") for r in rs])),
        ("Peak VRAM (MB)",     lambda rs: max([r.get("peak_vram_mb") or 0 for r in rs] + [0]) or None),
        ("Avg GPU Util (%)",   lambda rs: _avg([r.get("avg_gpu_util") for r in rs if r.get("avg_gpu_util") is not None])),
    ]
    for label, fn in metrics:
        row = [label]
        for m in models:
            val = fn(all_results[m])
            if val is None:
                row.append("—")
            elif label == "Total API Cost":
                row.append(f"${val:.4f}")
            else:
                row.append(f"{val:.1f}")
        perf.add_row(*row)
    console.print(perf)

    # E3-Score (efficiency-adjusted accuracy) — only shown when thinking tokens are present
    any_thinking = any(
        r.get("reasoning_tokens", 0) > 0
        for rs in all_results.values() for r in rs
    )
    if any_thinking:
        e3 = Table(box=box.ROUNDED, title="E3-Score — Efficiency-Adjusted Accuracy", show_lines=True)
        e3.add_column("Category", style="bold", min_width=22)
        for m in models:
            e3.add_column(_short(m), justify="center", min_width=16)

        for cat in categories:
            row = [cat]
            for m in models:
                rs = [r for r in all_results[m] if r["task"]["category"] == cat]
                scores = [
                    _e3_score(r["score"], r.get("reasoning_tokens"), cat)
                    for r in rs
                ]
                valid = [s for s in scores if s is not None]
                row.append(f"{sum(valid)/len(valid)*100:.0f}%" if valid else "—")
            e3.add_row(*row)

        # Weighted composite E3
        e3_total_row = ["[bold]E3 Composite ★[/bold]"]
        for m in models:
            rs = leaderboard_results(all_results[m])
            by_cat: dict = {}
            for r in rs:
                by_cat.setdefault(r["task"]["category"], []).append(r)
            w_sum, w_tot = 0.0, 0.0
            for cat, cat_rs in by_cat.items():
                valid = [
                    s for s in (
                        _e3_score(r["score"], r.get("reasoning_tokens"), cat)
                        for r in cat_rs
                    )
                    if s is not None
                ]
                if valid:
                    w = CATEGORY_WEIGHTS.get(cat.lower().strip(), 1.0)
                    w_sum += (sum(valid) / len(valid)) * w
                    w_tot += w
            e3_total_row.append(
                f"[bold]{w_sum/w_tot*100:.1f}%[/bold]" if w_tot > 0 else "—"
            )
        e3.add_row(*e3_total_row)
        console.print(e3)

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

    # Per-difficulty breakdown — only shown when tasks carry a 'difficulty' field
    all_difficulties = sorted({
        r["task"].get("difficulty", "").lower()
        for rs in all_results.values() for r in rs
        if r["task"].get("difficulty")
    })
    if all_difficulties:
        diff_table = Table(box=box.ROUNDED, title="Accuracy by Difficulty", show_lines=True)
        diff_table.add_column("Difficulty", style="bold", min_width=12)
        for m in models:
            diff_table.add_column(_short(m), justify="center", min_width=16)

        _DIFF_ORDER = ["easy", "medium", "hard", "expert"]
        sorted_diffs = sorted(all_difficulties, key=lambda d: (_DIFF_ORDER.index(d) if d in _DIFF_ORDER else 99))
        for diff in sorted_diffs:
            row = [diff.capitalize()]
            for m in models:
                rs = [r for r in all_results[m] if r["task"].get("difficulty", "").lower() == diff]
                if rs:
                    passed = sum(1 for r in rs if result_passed(r))
                    pct = sum(r["score"] for r in rs) / len(rs) * 100
                    row.append(f"{passed}/{len(rs)}  ({pct:.0f}%)")
                else:
                    row.append("—")
            diff_table.add_row(*row)
        console.print(diff_table)

    # Saturation warning: when any model scores above 85%, the benchmark
    # differentiates poorly — SWE-bench lesson.
    model_means = []
    for m in models:
        rs = leaderboard_results(all_results[m])
        if rs:
            model_means.append(sum(r["score"] for r in rs) / len(rs))
    if model_means and max(model_means) > 0.85:
        console.print(
            "\n[dim yellow]⚠  Scores near ceiling (>85%) — "
            "benchmark differentiates poorly at this level. "
            "Consider adding harder tasks (see BACKLOG.md §2.2).[/dim yellow]"
        )


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
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_record(result)) + "\n")

def _pair_ab_results(all_results: dict) -> dict:
    """Group ' [think]' / ' [no-think]' arm labels back to their base model.
    Returns {base_model: {"on": results, "off": results}} for complete pairs."""
    arms: dict = {}
    for label, results in all_results.items():
        if label.endswith(" [think]"):
            arms.setdefault(label[: -len(" [think]")], {})["on"] = results
        elif label.endswith(" [no-think]"):
            arms.setdefault(label[: -len(" [no-think]")], {})["off"] = results
    return {base: pair for base, pair in arms.items() if "on" in pair and "off" in pair}


def print_ab_thinking_summary(all_results: dict):
    """Per-model thinking-vs-no-thinking delta (--ab-thinking)."""
    pairs = _pair_ab_results(all_results)
    if not pairs:
        return

    t = Table(box=box.ROUNDED, title="Thinking A/B — harness-controlled", show_lines=True)
    t.add_column("Model", style="bold")
    t.add_column("Score (think)", justify="center")
    t.add_column("Score (no-think)", justify="center")
    t.add_column("Δ Score", justify="center")
    t.add_column("Think Tokens", justify="center")
    t.add_column("Δ Total ms", justify="center")

    for base, pair in pairs.items():
        on, off = pair["on"], pair["off"]
        score_on = _avg([r["score"] for r in on]) or 0.0
        score_off = _avg([r["score"] for r in off]) or 0.0
        delta = score_on - score_off
        delta_style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
        think_tokens = _avg([r.get("reasoning_tokens") for r in on])
        ms_on = _avg([r.get("total_ms") for r in on])
        ms_off = _avg([r.get("total_ms") for r in off])
        ms_delta = (
            f"{ms_on - ms_off:+.0f}" if ms_on is not None and ms_off is not None else "—"
        )
        t.add_row(
            base,
            f"{score_on * 100:.1f}%",
            f"{score_off * 100:.1f}%",
            f"[{delta_style}]{delta * 100:+.1f}%[/{delta_style}]",
            f"{think_tokens:.0f}" if think_tokens is not None else "—",
            ms_delta,
        )
    console.print(t)


def save_results(all_results: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON — full detail (same record shape as the JSONL stream)
    json_path = out / f"results_{ts}.json"
    payload: dict = {}
    for model, results in all_results.items():
        payload[model] = [to_record(r) for r in results]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"\nJSON → [cyan]{json_path}[/cyan]")

    # CSV — flat for spreadsheet comparison
    csv_path = out / f"results_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "task_id", "task_version", "category", "benchmark_tier",
                    "contamination_risk", "execution_surface", "source_signal",
                    "human_minutes_estimate", "criticisms_addressed", "scoring_type",
                    "score", "score_std", "pass_threshold", "passed", "tps", "ttft_ms", "total_ms",
                    "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
                    "api_cost", "sample_count", "agent_loop_progress_score",
                    "agent_loop_progress_passed", "agent_loop_progress_total",
                    "agent_loop_termination",
                    "score_detail"])
        for model, results in all_results.items():
            for r in results:
                task = r["task"]
                w.writerow([
                    model,
                    task["id"],
                    task.get("_version", ""),
                    task["category"],
                    task.get("benchmark_tier", ""),
                    task.get("contamination_risk", ""),
                    task.get("execution_surface", ""),
                    task.get("source_signal", task.get("_signal_source", "")),
                    task.get("human_minutes_estimate", ""),
                    _csv_value(task.get("criticisms_addressed", "")),
                    task.get("scoring", {}).get("type", ""),
                    r["score"],
                    r.get("score_std", ""),
                    r.get("pass_threshold", ""),
                    result_passed(r),
                    r.get("tps", ""),
                    r.get("ttft_ms", ""),
                    r.get("total_ms", ""),
                    r.get("prompt_tokens", ""),
                    r.get("completion_tokens", ""),
                    r.get("reasoning_tokens", ""),
                    r.get("total_tokens", ""),
                    r.get("api_cost", ""),
                    r.get("sample_count", ""),
                    r.get("agent_loop_progress_score", ""),
                    r.get("agent_loop_progress_passed", ""),
                    r.get("agent_loop_progress_total", ""),
                    r.get("agent_loop_termination", ""),
                    r.get("score_detail", ""),
                ])
    console.print(f"CSV  → [cyan]{csv_path}[/cyan]")


def save_html_report(all_results: dict, output_dir: str):
    """Generate a visual side-by-side HTML comparison of model outputs."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = out / f"report_{ts}.html"
    
    # Organize by task ID
    tasks_dict = {}
    models = list(all_results.keys())
    for model, results in all_results.items():
        for r in results:
            task_id = r["task"]["id"]
            if task_id not in tasks_dict:
                tasks_dict[task_id] = {"task": r["task"], "results": {}}
            tasks_dict[task_id]["results"][model] = r

    # Simple HTML generator
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>LLM Benchmark Report</title>",
        "<style>",
        "body { font-family: system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }",
        "h1 { color: #58a6ff; text-align: center; border-bottom: 1px solid #30363d; padding-bottom: 10px; }",
        ".task-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 30px; overflow: hidden; }",
        ".task-header { background: #21262d; padding: 15px 20px; border-bottom: 1px solid #30363d; }",
        ".task-title { margin: 0; color: #c9d1d9; font-size: 1.2em; }",
        ".task-category { display: inline-block; background: #1f6feb; color: #ffffff; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 10px; }",
        ".prompt { background: #0d1117; padding: 10px; border-radius: 4px; border: 1px solid #30363d; white-space: pre-wrap; font-family: monospace; font-size: 0.9em; margin-top: 10px; color: #8b949e; }",
        ".grid { display: flex; overflow-x: auto; padding: 20px; gap: 20px; }",
        ".col { flex: 1; min-width: 350px; max-width: 600px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 15px; }",
        ".model-name { margin-top: 0; color: #8a4baf; border-bottom: 1px solid #30363d; padding-bottom: 10px; font-size: 1.1em; }",
        ".metrics { display: flex; gap: 10px; margin-bottom: 15px; font-size: 0.85em; }",
        ".metric { background: #21262d; padding: 4px 8px; border-radius: 4px; color: #8b949e; }",
        ".score-pass { color: #3fb950; font-weight: bold; }",
        ".score-fail { color: #f85149; font-weight: bold; }",
        ".reasoning { background: #161b22; border-left: 3px solid #8b949e; padding: 10px; margin-bottom: 15px; font-size: 0.9em; color: #8b949e; font-style: italic; white-space: pre-wrap; }",
        ".response { white-space: pre-wrap; font-size: 0.95em; line-height: 1.5; }",
        "</style></head><body>",
        "<h1>LLM Benchmark Visual Report</h1>"
    ]

    for task_id, data in tasks_dict.items():
        task = data["task"]
        html.append(f"<div class='task-card'>")
        html.append(f"<div class='task-header'>")
        html.append(f"<h2 class='task-title'><span class='task-category'>{escape(task.get('category', 'misc'))}</span>{escape(task_id)}</h2>")
        html.append(f"<div class='prompt'>{escape(task.get('prompt', ''))}</div>")
        html.append("</div>")
        html.append("<div class='grid'>")
        
        for model in models:
            r = data["results"].get(model)
            html.append("<div class='col'>")
            html.append(f"<h3 class='model-name'>{escape(model)}</h3>")
            if not r:
                html.append("<div style='color: #8b949e;'>No result</div></div>")
                continue
            
            score_cls = "score-pass" if result_passed(r) else "score-fail"
            score_text = f"Score: {r.get('score', 0):.1f}"
            if r.get("score_detail"):
                score_text += f" ({r['score_detail'][:50]})"
                
            html.append("<div class='metrics'>")
            html.append(f"<span class='metric {score_cls}'>{escape(score_text)}</span>")
            if r.get("tps"): html.append(f"<span class='metric'>{r['tps']:.1f} t/s</span>")
            if r.get("ttft_ms"): html.append(f"<span class='metric'>{r['ttft_ms']:.0f}ms ttft</span>")
            html.append("</div>")
            
            if r.get("reasoning_preview"):
                reasoning = escape(r["reasoning_preview"])
                html.append(f"<div class='reasoning'><strong>Reasoning:</strong><br>{reasoning}</div>")

            response = escape(str(r.get("response", "")))
            html.append(f"<div class='response'>{response}</div>")
            html.append("</div>")
            
        html.append("</div></div>")

    html.append("</body></html>")
    
    html_path.write_text("\n".join(html), encoding="utf-8")
    console.print(f"HTML → [cyan]{html_path}[/cyan]")
