"""
benchmark/auditor.py — training-data contamination probe (BACKLOG 4.3).

Method from the OpenAI SWE-bench Verified post-mortem: send each coding
task's *id alone* — never the problem statement — and check whether the
model reproduces the solution's identifying signals (the function names
the scoring harness calls). A model that produces the right function
signature from an opaque task id has very likely seen the task in its
training data.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from rich import box
from rich.table import Table

from benchmark.console import make_console
from benchmark.runner import ModelRunner

console = make_console()

# A task is flagged when the model reproduces at least this share of its
# identifying signals from the id alone.
FLAG_THRESHOLD = 0.5

# Identifiers that appear in test_code but are not solution functions.
_COMMON_CALLS = {
    "print", "len", "range", "sorted", "set", "list", "dict", "tuple", "str",
    "int", "float", "bool", "abs", "sum", "min", "max", "isinstance",
    "enumerate", "zip", "map", "filter", "round", "repr", "type", "all",
    "any", "getattr", "hasattr", "format", "open", "iter", "next",
}


def expected_signals(task: dict) -> list[str]:
    """Function names the scoring harness calls — the contamination tells."""
    test_code = task.get("scoring", {}).get("test_code", "")
    seen: set[str] = set()
    signals: list[str] = []
    for name in re.findall(r"\b([A-Za-z_]\w*)\s*\(", test_code):
        if name in _COMMON_CALLS or name in seen:
            continue
        seen.add(name)
        signals.append(name)
    return signals


def probe_task(task: dict) -> dict:
    """The probe deliberately contains the task id and nothing else."""
    return {
        "id": f"audit_{task['id']}",
        "category": "audit",
        "prompt": (
            "You are an LLM benchmark suite's contamination probe. "
            f"A benchmark contains a Python coding task whose id is '{task['id']}'. "
            "Without being shown the problem statement, write the complete Python "
            "solution you believe this exact task expects. If you do not recognize "
            "the task id, reply with exactly: UNKNOWN"
        ),
        "max_tokens": 800,
    }


def audit_contamination(
    model_pairs: list,
    tasks: list[dict],
    bench_config: dict,
    no_autoload: bool = True,
) -> dict:
    """
    Probe every model with every auditable task (those whose scoring carries
    test_code). Returns {model_id: [row, ...]} where each row records the
    expected signals, which ones the model reproduced, and whether the task
    is flagged.
    """
    auditable = [t for t in tasks if expected_signals(t)]
    if not auditable:
        return {}

    # One generation per probe — variance does not matter for a tell.
    probe_config = dict(bench_config, runs_per_task=1)

    report: dict = {}
    for model_info, backend in model_pairs:
        runner = ModelRunner(backend, model_info.id, probe_config)
        if backend.config.get("auto_load", False) and not no_autoload:
            runner.ensure_model_loaded()

        rows = []
        for task in auditable:
            signals = expected_signals(task)
            raw = runner.run_task(probe_task(task))
            response = raw.get("response") or ""
            matched = [s for s in signals if s in response]
            match_rate = len(matched) / len(signals)
            rows.append({
                "task_id": task["id"],
                "signals": signals,
                "matched": matched,
                "match_rate": match_rate,
                "claimed_unknown": "UNKNOWN" in response[:80].upper(),
                "flagged": match_rate >= FLAG_THRESHOLD,
                "error": raw.get("error"),
            })
            icon = "[red]⚑[/red]" if rows[-1]["flagged"] else "[green]·[/green]"
            console.print(
                f"  {icon} {task['id']:32s} "
                f"[dim]{len(matched)}/{len(signals)} signal(s) reproduced[/dim]"
            )
        report[model_info.id] = rows
    return report


def print_audit_report(report: dict) -> None:
    if not report:
        console.print("[yellow]No auditable tasks (none carry code_exec test_code).[/yellow]")
        return

    t = Table(box=box.ROUNDED, title="Contamination Audit — task id only, no problem statement", show_lines=True)
    t.add_column("Model", style="bold")
    t.add_column("Flagged", justify="center")
    t.add_column("Claimed unknown", justify="center")
    t.add_column("Flagged tasks", style="dim")

    for model, rows in report.items():
        flagged = [r for r in rows if r["flagged"]]
        unknown = sum(1 for r in rows if r["claimed_unknown"])
        style = "red" if flagged else "green"
        t.add_row(
            model,
            f"[{style}]{len(flagged)}/{len(rows)}[/{style}]",
            f"{unknown}/{len(rows)}",
            ", ".join(r["task_id"] for r in flagged) or "—",
        )
    console.print(t)
    console.print(
        "[dim]A flagged task means the model reproduced ≥"
        f"{FLAG_THRESHOLD:.0%} of the solution's function names from the "
        "opaque task id — treat its benchmark score as contaminated.[/dim]"
    )


def save_audit_report(report: dict, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"audit_{ts}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"Audit → [cyan]{path}[/cyan]")
    return path
