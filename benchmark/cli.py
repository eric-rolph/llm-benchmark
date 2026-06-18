"""
benchmark/cli.py — entry point for the LLM Benchmark Suite (`llm-bench`).

Supported backends (configure in config.yaml):
  lm_studio   — LM Studio local server  (http://localhost:1234)
  ollama      — Ollama                  (http://localhost:11434)
  llamacpp    — llama.cpp server        (http://localhost:8080)
  vllm        — vLLM                    (http://localhost:8000)
  sglang      — SGLang                  (http://localhost:30000)
  tgi         — Text Generation Inference
  tensorrt    — TensorRT-LLM
  ktransformers — KTransformers
  generic_openai — Any OpenAI-compatible server

Quick start (llm-bench, or python run.py from a checkout):
  llm-bench                          # auto-discover models, run all tasks
  llm-bench --discover               # probe backends and list found models
  llm-bench --dry-run                # validate task files + check backends, no inference
  llm-bench --model "qwen3:8b"       # single model (all categories)
  llm-bench --model "modelA" "modelB"# multiple models (for arena mode)
  llm-bench --backend ollama         # only Ollama models
  llm-bench --category math          # single category
  llm-bench --task capital_france    # single task by ID
  llm-bench --task task_a task_b     # multiple explicit task IDs
  llm-bench --no-autoload            # skip LM Studio model-load attempt
  llm-bench --allow-code-exec        # enable code_exec scoring (runs model-generated Python)
  llm-bench --ci-threshold 0.8       # exit 1 if overall score < 80%  (CI integration)
  llm-bench --html-report            # generate interactive HTML visual report
  llm-bench --arena                  # ELO arena: pairwise model competition with LLM judge
  llm-bench --compare old new        # compare two saved JSON/JSONL result files
  llm-bench --limit 5                # smoke-test: first 5 tasks per category
  llm-bench --resume                 # skip tasks already in the most recent results JSONL
  llm-bench --max-api-cost 5.00      # stop launching new uncached tasks after provider-reported API cost reaches this USD cap
  llm-bench --exclude-before 2026-06-01  # only tasks introduced on/after a date (contamination control)
  llm-bench --ab-thinking            # run each task with thinking on vs off, report the delta (Ollama)
  llm-bench --audit-contamination    # probe models with task IDs only, flag memorised solutions
  llm-bench --judge-model qwen3:8b   # enable LLM judge with a local model (CI-friendly)
  llm-bench --judge-model gpt-4o --judge-base-url https://api.openai.com/v1 --judge-api-key sk-…
"""
from __future__ import annotations

import argparse
import datetime
import sys
from collections import defaultdict
from pathlib import Path

from benchmark.arena import run_arena, print_arena_leaderboard, save_arena_results
from benchmark.compare import compare_result_files
from benchmark.console import make_console
from benchmark.evaluation import leaderboard_results, result_passed
from benchmark.loader import available_categories, filter_introduced_since, load_tasks
from benchmark.reporter import print_ab_thinking_summary, print_report, save_results, save_html_report
from benchmark.session import (
    build_judge,
    discover_models,
    load_cached_records,
    load_config,
    print_discovery_table,
    resolve_api_cost_budget,
    run_model,
)

console = make_console()


def _early_tasks_dir() -> str | None:
    """Peek at --tasks-dir before the parser exists — the --category choices
    list depends on it, and argparse can't bootstrap itself."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--tasks-dir" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--tasks-dir="):
            return arg.split("=", 1)[1]
    return None


def _print_model_summary(model_results: list[dict]) -> None:
    passed    = sum(1 for r in model_results if result_passed(r))
    tps_vals  = [r["tps"] for r in model_results if r.get("tps")]
    tok_vals  = [r["completion_tokens"] for r in model_results if r.get("completion_tokens")]
    avg_tps   = sum(tps_vals) / len(tps_vals) if tps_vals else 0
    avg_tok   = sum(tok_vals) / len(tok_vals) if tok_vals else None
    score_pct = passed / len(model_results) * 100 if model_results else 0
    tok_str   = f"  Avg Tokens: {avg_tok:.0f}" if avg_tok else ""
    console.print(
        f"\n  [bold]Score: {passed}/{len(model_results)} "
        f"({score_pct:.0f}%)  "
        f"Avg TPS: {avg_tps:.1f}"
        f"{tok_str}[/bold]\n"
    )


def _filter_tasks_by_ids(tasks: list[dict], task_ids: list[str]) -> list[dict]:
    requested = set(task_ids)
    filtered = [task for task in tasks if task.get("id") in requested]
    found = {task.get("id") for task in filtered}
    missing = [task_id for task_id in task_ids if task_id not in found]
    if missing:
        raise ValueError(f"Task ID(s) not found: {', '.join(missing)}")
    return filtered


def main():
    tasks_dir = _early_tasks_dir()
    categories = available_categories(tasks_dir)

    parser = argparse.ArgumentParser(
        description="Automated LLM benchmark — LM Studio · Ollama · llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",         default="config.yaml",  help="Path to config file")
    parser.add_argument("--model",          nargs="+", metavar="MODEL",
                        help="Benchmark only these model ID(s) (matched against any backend; use multiple for arena)")
    parser.add_argument("--backend",        help="Restrict to one backend type (lm_studio | ollama | llamacpp)")
    parser.add_argument("--category",       choices=categories,      help="Run only this task category")
    parser.add_argument("--task",           nargs="+",
                        help="Run only the specified task ID(s)")
    parser.add_argument("--tasks-dir",      default=None, metavar="DIR",
                        help="Load tasks from this directory instead of the bundled tasks/")
    parser.add_argument("--output",         default="results",       help="Results output directory")
    parser.add_argument("--no-autoload",    action="store_true",     help="Skip LM Studio model-load attempt")
    parser.add_argument("--discover",       action="store_true",     help="Probe backends and print discovered models, then exit")
    parser.add_argument("--list-models",    action="store_true",     help="Alias for --discover")
    parser.add_argument("--dry-run",        action="store_true",     help="Validate task files and check backend connectivity, no inference")
    parser.add_argument("--allow-code-exec",action="store_true",     help="Enable code_exec scoring (runs model-generated Python locally — review tasks first)")
    parser.add_argument("--html-report",    action="store_true",     help="Generate an interactive HTML visual report of the results")
    parser.add_argument("--arena",          action="store_true",     help="Arena mode: pairwise ELO competition between all discovered models")
    parser.add_argument("--compare",        nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="Compare two saved JSON/JSONL result files and exit")
    parser.add_argument("--compare-top",    type=int, default=10, metavar="N",
                        help="Number of largest task deltas to show with --compare")
    parser.add_argument("--ci-threshold",   type=float, default=None,metavar="RATIO",
                        help="Exit with code 1 if overall score ratio is below this (e.g. 0.8 = 80%%)")
    parser.add_argument("--limit",          type=int,   default=None, metavar="N",
                        help="Run only the first N tasks per category (smoke test / quick iteration)")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip (model, task) pairs already in the most recent results JSONL (continue interrupted run)")
    parser.add_argument("--max-api-cost",   type=float, default=None, metavar="USD",
                        help="Stop launching new uncached tasks after provider-reported api_cost reaches this run budget")
    parser.add_argument("--exclude-before", default=None, metavar="DATE",
                        help="Run only tasks introduced on/after this date (YYYY-MM-DD); tasks without an 'introduced' tag are excluded (contamination control)")
    parser.add_argument("--ab-thinking",    action="store_true",
                        help="Run every task twice — thinking on vs off — and report the per-model delta (backends with a thinking toggle, i.e. Ollama)")
    parser.add_argument("--audit-contamination", action="store_true",
                        help="Probe each model with task IDs only (no problem statements) and flag tasks whose solutions the model reproduces — no scoring run")
    # Judge CLI flags — bypass config.yaml and the interactive TTY prompt
    parser.add_argument("--judge-model",    default=None, metavar="MODEL",
                        help="Enable LLM judge with this model (bypasses interactive prompt; use a discovered model ID or an external one with --judge-base-url)")
    parser.add_argument("--judge-api-key",  default=None, metavar="KEY",
                        help="API key for an external judge model (falls back to $OPENAI_API_KEY)")
    parser.add_argument("--judge-base-url", default=None, metavar="URL",
                        help="API base URL for an external judge model (required when --judge-model is not a discovered local model)")
    args = parser.parse_args()

    if args.compare:
        compare_result_files(args.compare[0], args.compare[1], top_n=args.compare_top)
        return

    config = load_config(args.config)
    bench_config = config.get("benchmark", {})
    try:
        api_cost_budget = resolve_api_cost_budget(bench_config, args.max_api_cost)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    # ── discover ──────────────────────────────────────────────────────────
    console.print(f"\n[bold]LLM Benchmark Suite[/bold]  [dim]Probing backends…[/dim]")

    all_pairs = discover_models(config, backend_filter=args.backend)

    if args.discover or args.list_models:
        print_discovery_table(all_pairs)
        return

    # ── dry-run: validate only ────────────────────────────────────────────
    if args.dry_run:
        console.print("\n[bold]Dry-run mode[/bold] — validating task files…")
        try:
            tasks_all = load_tasks(validate=True, expand_datasets=False, tasks_dir=args.tasks_dir)
            cats = sorted(set(t["category"] for t in tasks_all))
            console.print(f"  [green]✓[/green]  {len(tasks_all)} tasks loaded across {len(cats)} categories: {cats}")
        except (ValueError, FileNotFoundError) as e:
            console.print(f"  [red]✗  Task validation failed:[/red] {e}")
            sys.exit(1)
        console.print(f"  [green]✓[/green]  Backends: {len(all_pairs)} model(s) discoverable")
        console.print("\n[green]Dry run passed.[/green]")
        return

    # Filter to --model if specified
    if args.model:
        all_pairs = [(m, b) for m, b in all_pairs
                     if any(pat in m.id for pat in args.model)]
        if not all_pairs:
            console.print(f"[red]Model(s) {args.model} not found in any enabled backend.[/red]")
            console.print("[dim]Run --discover to see available models.[/dim]")
            sys.exit(1)

    if not all_pairs:
        console.print("[red]No models to benchmark. Enable a backend in config.yaml or use --model.[/red]")
        console.print("[dim]Run --discover to see what is available.[/dim]")
        sys.exit(1)

    # ── tasks ─────────────────────────────────────────────────────────────
    tasks = load_tasks(args.category, tasks_dir=args.tasks_dir)
    if args.task:
        try:
            tasks = _filter_tasks_by_ids(tasks, args.task)
        except ValueError as exc:
            console.print(f"[red]{exc}. Use --dry-run to list all task IDs.[/red]")
            sys.exit(1)
    if not tasks:
        console.print("[red]No tasks loaded. Check the tasks/ directory.[/red]")
        sys.exit(1)

    # Apply --exclude-before: fresh-task subset for contamination control
    if args.exclude_before:
        try:
            cutoff = datetime.date.fromisoformat(args.exclude_before)
        except ValueError:
            console.print(f"[red]--exclude-before expects YYYY-MM-DD, got {args.exclude_before!r}.[/red]")
            sys.exit(1)
        total = len(tasks)
        tasks = filter_introduced_since(tasks, cutoff)
        console.print(f"[dim]--exclude-before {cutoff}: {len(tasks)}/{total} task(s) introduced on/after the cutoff[/dim]")
        if not tasks:
            console.print("[red]No tasks introduced on/after the cutoff. Tag tasks with 'introduced: YYYY-MM-DD'.[/red]")
            sys.exit(1)

    # Apply --limit: keep only first N tasks per category
    if args.limit:
        seen: dict = defaultdict(int)
        limited = []
        for t in tasks:
            cat = t.get("category", "")
            if seen[cat] < args.limit:
                limited.append(t)
                seen[cat] += 1
        tasks = limited
        console.print(f"[dim]--limit {args.limit}: {len(tasks)} tasks (first {args.limit} per category)[/dim]")

    # ── contamination audit: probe-only mode, no benchmark run ────────────
    if args.audit_contamination:
        from benchmark.auditor import audit_contamination, print_audit_report, save_audit_report
        console.print("\n[bold]Contamination audit[/bold] — probing with task IDs only…\n")
        report = audit_contamination(
            all_pairs, tasks, bench_config, no_autoload=args.no_autoload
        )
        print_audit_report(report)
        if report:
            save_audit_report(report, args.output)
        return

    # ── LLM judge: CLI flag > config > interactive prompt ─────────────────
    judge_client, judge_model = build_judge(args, config, all_pairs, tasks)

    # ── security warning ──────────────────────────────────────────────────
    if args.allow_code_exec:
        console.print("\n[bold red]⚠️ WARNING: --allow-code-exec is enabled. Untrusted LLM code will be executed locally.[/bold red]")
        if sys.platform == "win32":
            console.print("[bold red]⚠️ SECURITY: You are on Windows. Memory limits (RLIMIT_AS) are NOT supported. A malicious model could consume all system RAM.[/bold red]\n")

    cats = sorted(set(t["category"] for t in tasks))
    console.print(
        f"[dim]{len(tasks)} tasks · categories: {cats} · "
        f"{len(all_pairs)} model(s)[/dim]\n"
    )
    if api_cost_budget:
        console.print(f"[dim]API cost cap: ${api_cost_budget.limit:.4f} for newly executed tasks[/dim]\n")

    # ── Arena mode ─────────────────────────────────────────────────────────
    if args.arena:
        if not judge_client:
            # Auto-enable judge for arena mode using first discovered model
            judge_model = all_pairs[0][0].id
            judge_client = all_pairs[0][1].get_openai_client()
            console.print(f"[dim]Arena mode: auto-enabled judge — model: {judge_model}[/dim]")
        players = run_arena(
            model_pairs=all_pairs,
            tasks=tasks,
            bench_config=bench_config,
            judge_client=judge_client,
            judge_model=judge_model,
            no_autoload=args.no_autoload,
            api_cost_budget=api_cost_budget,
        )
        print_arena_leaderboard(players)
        save_arena_results(players, args.output)
        return

    all_results: dict = {}
    stop_for_budget = False

    # Incremental JSONL for crash safety (one line per task, written immediately)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    resume = bench_config.get("resume", False) or args.resume
    cached_records: dict = {}
    jsonl_path = None
    if resume:
        cached_records, jsonl_path = load_cached_records(out_dir)
        if jsonl_path is not None:
            console.print(f"[dim]Resuming from {jsonl_path.name} — {len(cached_records)} compatible cached result(s)[/dim]")
        else:
            console.print("[dim]--resume: no existing JSONL found — starting fresh[/dim]")
    if jsonl_path is None:
        jsonl_path = out_dir / f"results_{ts}.jsonl"
    console.print(f"[dim]Streaming results → {jsonl_path}[/dim]\n")

    for model_info, backend in all_pairs:
        if stop_for_budget:
            break
        ab_capable = getattr(backend, "supports_thinking_ab", False)
        if args.ab_thinking and not ab_capable:
            console.print(
                f"[yellow]--ab-thinking: {backend.name} has no thinking toggle — "
                f"running {model_info.id} once, without arms.[/yellow]"
            )
        if args.ab_thinking and ab_capable:
            arms = [(" [think]", True), (" [no-think]", False)]
        else:
            arms = [("", None)]

        for arm_suffix, think_value in arms:
            if api_cost_budget and api_cost_budget.exhausted:
                stop_for_budget = True
                break
            run_label = f"{model_info.id}{arm_suffix}"
            arm_tasks = (
                tasks if think_value is None
                else [dict(t, thinking=think_value) for t in tasks]
            )
            label = f"{run_label}  [dim]({backend.name})[/dim]"
            console.rule(f"[bold cyan]{label}[/bold cyan]")

            model_results = run_model(
                model_info=model_info,
                backend=backend,
                tasks=arm_tasks,
                bench_config=bench_config,
                cached_records=cached_records,
                jsonl_path=jsonl_path,
                allow_code_exec=args.allow_code_exec,
                no_autoload=args.no_autoload,
                judge_client=judge_client,
                judge_model=judge_model,
                result_label=run_label if arm_suffix else None,
                api_cost_budget=api_cost_budget,
            )
            all_results[run_label] = model_results

            _print_model_summary(model_results)
            if api_cost_budget and api_cost_budget.exhausted:
                stop_for_budget = True
                break

        if stop_for_budget and api_cost_budget:
            console.print(
                f"[yellow]Stopped because API cost budget is exhausted "
                f"(${api_cost_budget.spent:.4f} / ${api_cost_budget.limit:.4f}).[/yellow]\n"
            )

    print_report(all_results, expected_tasks=tasks)
    if args.ab_thinking:
        print_ab_thinking_summary(all_results)
    save_results(all_results, args.output)
    if args.html_report:
        save_html_report(all_results, args.output)

    # ── CI threshold check ────────────────────────────────────────────────
    if args.ci_threshold is not None:
        all_scored = [r for rs in all_results.values() for r in rs]
        scored_for_ci = leaderboard_results(all_scored)
        if all_scored and not scored_for_ci:
            console.print("[yellow]CI threshold: no leaderboard-core rows; falling back to all scored tasks.[/yellow]")
            scored_for_ci = all_scored
        if scored_for_ci:
            ratio = sum(r["score"] for r in scored_for_ci) / len(scored_for_ci)
            pct = ratio * 100
            if ratio < args.ci_threshold:
                console.print(
                    f"\n[red bold]CI FAILED[/red bold]  "
                    f"Score {pct:.1f}% < threshold {args.ci_threshold * 100:.0f}%"
                )
                sys.exit(1)
            else:
                console.print(
                    f"\n[green bold]CI PASSED[/green bold]  "
                    f"Score {pct:.1f}% ≥ threshold {args.ci_threshold * 100:.0f}%"
                )


if __name__ == "__main__":
    main()
