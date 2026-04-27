"""
run.py — entry point for the LLM Benchmark Suite.

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

Quick start:
  python run.py                          # auto-discover models, run all tasks
  python run.py --discover               # probe backends and list found models
  python run.py --dry-run                # validate task files + check backends, no inference
  python run.py --model "qwen3:8b"       # single model (all categories)
  python run.py --model "modelA" "modelB"# multiple models (for arena mode)
  python run.py --backend ollama         # only Ollama models
  python run.py --category math          # single category
  python run.py --task capital_france    # single task by ID
  python run.py --no-autoload            # skip LM Studio model-load attempt
  python run.py --allow-code-exec        # enable code_exec scoring (runs model-generated Python)
  python run.py --ci-threshold 0.8       # exit 1 if overall score < 80%  (CI integration)
  python run.py --html-report            # generate interactive HTML visual report
  python run.py --arena                  # ELO arena: pairwise model competition with LLM judge
  python run.py --compare old new        # compare two saved JSON/JSONL result files
  python run.py --limit 5                # smoke-test: first 5 tasks per category
  python run.py --resume                 # skip tasks already in the most recent results JSONL
"""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
from pathlib import Path

import yaml
from rich.table import Table
from rich import box

from benchmark.backends import create_backend, discover_all_models
from benchmark.backends.base import ModelInfo
from benchmark.console import make_console
from benchmark.loader import load_tasks
from benchmark.reporter import print_report, print_task_result, save_results, append_jsonl, save_html_report
from benchmark.runner import ModelRunner
from benchmark.scorer import score_response, score_pass_at_k
from benchmark.arena import run_arena, print_arena_leaderboard, save_arena_results
from benchmark.compare import compare_result_files
from benchmark.utils import task_fingerprint

console = make_console()

TASK_DIR = Path(__file__).parent / "tasks"


def _cache_key(model_id: str, task: dict) -> tuple[str, str, str, str]:
    """Key cached results by model, task id, declared version, and task content."""
    return (
        model_id,
        task["id"],
        str(task.get("_version", "")),
        task_fingerprint(task),
    )


def _record_cache_key(record: dict) -> tuple[str, str, str, str]:
    """Build the resume key stored in JSONL records."""
    return (
        record.get("model_id", record.get("model", "")),
        record.get("task_id", ""),
        str(record.get("task_version", "")),
        record.get("task_hash", ""),
    )


def _hydrate_cached_result(task: dict, record: dict) -> dict:
    """Convert a cached JSONL row back into a scored result for reporting."""
    return {
        "task_id": task["id"],
        "task": task,
        "response": record.get("response_preview", ""),
        "error": None,
        "score": float(record.get("score", 0.0)),
        "max_score": 1.0,
        "score_detail": record.get("score_detail", ""),
        "tps": record.get("tps"),
        "ttft_ms": record.get("ttft_ms"),
        "total_ms": record.get("total_ms"),
        "completion_tokens": record.get("completion_tokens"),
        "reasoning_tokens": record.get("reasoning_tokens", 0),
        "peak_vram_mb": record.get("peak_vram_mb"),
        "avg_gpu_util": record.get("avg_gpu_util"),
        "backend": record.get("backend", "?"),
        "model_id": record.get("model_id", record.get("model", "?")),
        "logprob_detail": record.get("logprob_detail"),
        "hf_generation_config": record.get("hf_generation_config"),
    }


def _run_model(
    model_info: ModelInfo,
    backend,
    tasks: list[dict],
    bench_config: dict,
    cached_records: dict,
    jsonl_path: Path,
    allow_code_exec: bool,
    no_autoload: bool,
    judge_client,
    judge_model: str | None,
) -> list[dict]:
    """Run all tasks for one model and return the list of scored results."""
    runner = None
    model_results: list = []
    for task in tasks:
        cache_key = _cache_key(model_info.id, task)
        if cache_key in cached_records:
            console.print(f"  [dim]\u21a9  {task['id']:40s}  (cached \u2014 skipped)[/dim]")
            model_results.append(_hydrate_cached_result(task, cached_records[cache_key]))
            continue

        if runner is None:
            runner = ModelRunner(backend, model_info.id, bench_config)
            if backend.config.get("auto_load", False) and not no_autoload:
                runner.ensure_model_loaded()

        scoring_type = task.get("scoring", {}).get("type")
        if scoring_type == "pass_at_k":
            k = task["scoring"].get("k", bench_config.get("runs_per_task", 3))
            n = task["scoring"].get(
                "n",
                task["scoring"].get("samples", bench_config.get("runs_per_task", k)),
            )
            n = max(int(k), int(n))
            raws = runner.run_task_k(task, n)
            scored = score_pass_at_k(
                task, raws,
                allow_code_exec=allow_code_exec,
                judge_client=judge_client,
                judge_model=judge_model,
            )
        else:
            raw = runner.run_task(task)
            scored = score_response(
                task, raw,
                allow_code_exec=allow_code_exec,
                judge_client=judge_client,
                judge_model=judge_model,
            )
            # When runs_per_task > 1, score every individual run and record variance.
            all_runs = raw.pop("_all_runs", None)
            if all_runs and len(all_runs) > 1:
                per_run_scores = []
                for r in all_runs:
                    if not r.get("error"):
                        s = score_response(
                            task, r,
                            allow_code_exec=allow_code_exec,
                            judge_client=judge_client,
                            judge_model=judge_model,
                        )
                        per_run_scores.append(s["score"])
                if per_run_scores:
                    scored["score"] = statistics.mean(per_run_scores)
                    scored["score_std"] = (
                        statistics.stdev(per_run_scores) if len(per_run_scores) > 1 else 0.0
                    )

        scored["model_id"] = model_info.id
        model_results.append(scored)
        print_task_result(scored)
        append_jsonl(scored, jsonl_path)

    return model_results


def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        console.print(f"[red]Config not found: {cfg_path}[/red]")
        sys.exit(1)
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _available_categories() -> list[str]:
    return sorted(p.stem for p in TASK_DIR.glob("*.yaml"))


def _discover_models(config: dict, backend_filter: str | None = None) -> list[tuple[ModelInfo, object]]:
    """
    Returns list of (ModelInfo, Backend instance) pairs for all enabled backends.
    backend_filter: restrict to a specific backend type name.
    """
    pairs: list[tuple[ModelInfo, object]] = []
    backends_cfg: dict = config.get("backends", {})
    manual_ids: list[str] = config.get("models", [])

    for backend_type, backend_cfg in backends_cfg.items():
        if not backend_cfg.get("enabled", False):
            continue
        if backend_filter and backend_filter.lower() not in (backend_type, backend_cfg.get("name", "").lower()):
            continue

        try:
            backend = create_backend(backend_type, backend_cfg)
        except ValueError as e:
            console.print(f"[yellow]  Config error: {e}[/yellow]")
            continue

        if not backend.is_available():
            console.print(f"  [dim]{backend_cfg.get('name', backend_type)}: not reachable ({backend_cfg.get('base_url', '')})[/dim]")
            continue

        if backend_cfg.get("auto_discover", True):
            discovered = backend.discover_models()
        else:
            discovered = []

        # Apply manual model filter if set
        if manual_ids:
            discovered = [m for m in discovered if m.id in manual_ids]
            # Also add any manual models not found by discovery (direct specification)
            found_ids = {m.id for m in discovered}
            for mid in manual_ids:
                if mid not in found_ids:
                    discovered.append(ModelInfo(id=mid, name=mid, backend_name=backend.name))

        for m in discovered:
            pairs.append((m, backend))

    # If no backends are enabled, fall back to manual list with no backend
    if not pairs and manual_ids and not backend_filter:
        console.print("[yellow]No enabled backends are reachable. Manual model list requires a reachable backend.[/yellow]")

    return pairs


def _print_discovery_table(pairs: list[tuple[ModelInfo, object]]) -> None:
    if not pairs:
        console.print("[yellow]No models found. Enable a backend in config.yaml and start the server.[/yellow]")
        return

    t = Table(title="Discovered Models", box=box.ROUNDED, show_lines=False)
    t.add_column("Backend",    style="cyan",  no_wrap=True)
    t.add_column("Model ID",   style="white", no_wrap=False)
    t.add_column("Size",       style="dim",   justify="right")
    t.add_column("Details",    style="dim")

    for m, _ in pairs:
        size_str = f"{m.size_bytes / 1e9:.1f} GB" if m.size_bytes else ""
        details_parts = []
        if m.details.get("parameter_size"):
            details_parts.append(m.details["parameter_size"])
        if m.details.get("quantization_level"):
            details_parts.append(m.details["quantization_level"])
        if m.details.get("family"):
            details_parts.append(m.details["family"])
        t.add_row(m.backend_name, m.id, size_str, "  ".join(details_parts))

    console.print(t)
    console.print(f"\n[dim]Total: {len(pairs)} model(s)[/dim]\n")


def main():
    categories = _available_categories()

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
    parser.add_argument("--task",           help="Run only a single task by ID")
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
    args = parser.parse_args()

    if args.compare:
        compare_result_files(args.compare[0], args.compare[1], top_n=args.compare_top)
        return

    config = _load_config(args.config)
    bench_config = config.get("benchmark", {})

    # ── discover ──────────────────────────────────────────────────────────
    console.print(f"\n[bold]LLM Benchmark Suite[/bold]  [dim]Probing backends…[/dim]")

    all_pairs = _discover_models(config, backend_filter=args.backend)

    if args.discover or args.list_models:
        _print_discovery_table(all_pairs)
        return

    # ── dry-run: validate only ────────────────────────────────────────────
    if args.dry_run:
        console.print("\n[bold]Dry-run mode[/bold] — validating task files…")
        try:
            tasks_all = load_tasks(validate=True, expand_datasets=False)
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
    # ── LLM judge setup ────────────────────────────────────────────────────────
    judge_cfg = config.get("judge", {})
    judge_client = None
    judge_model: str | None = None
    if judge_cfg.get("enabled") and all_pairs:
        judge_model = judge_cfg.get("model") or all_pairs[0][0].id
        judge_client = all_pairs[0][1].get_openai_client()
        console.print(f"[dim]LLM judge enabled — model: {judge_model}[/dim]")
        
    # ── security warning ──────────────────────────────────────────────────────
    if args.allow_code_exec:
        console.print("\n[bold red]⚠️ WARNING: --allow-code-exec is enabled. Untrusted LLM code will be executed locally.[/bold red]")
        if sys.platform == "win32":
            console.print("[bold red]⚠️ SECURITY: You are on Windows. Memory limits (RLIMIT_AS) are NOT supported. A malicious model could consume all system RAM.[/bold red]\n")
    # ── tasks ─────────────────────────────────────────────────────────────
    tasks = load_tasks(args.category)
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            console.print(f"[red]Task '{args.task}' not found. Use --dry-run to list all task IDs.[/red]")
            sys.exit(1)
    if not tasks:
        console.print("[red]No tasks loaded. Check the tasks/ directory.[/red]")
        sys.exit(1)

    # Apply --limit: keep only first N tasks per category
    if args.limit:
        from collections import defaultdict
        seen: dict = defaultdict(int)
        limited = []
        for t in tasks:
            cat = t.get("category", "")
            if seen[cat] < args.limit:
                limited.append(t)
                seen[cat] += 1
        tasks = limited
        console.print(f"[dim]--limit {args.limit}: {len(tasks)} tasks (first {args.limit} per category)[/dim]")

    # ── interactive judge setup ───────────────────────────────────────────────
    # Prompt once when the judge is disabled but the task list contains judge
    # tasks, and we're running interactively (not piped / CI).
    if not judge_cfg.get("enabled") and sys.stdin.isatty():
        _judge_types = {"llm_judge", "rubric_judge"}
        _judge_tasks = [t for t in tasks if t.get("scoring", {}).get("type") in _judge_types]
        if _judge_tasks:
            _examples = ", ".join(t["id"] for t in _judge_tasks[:3])
            if len(_judge_tasks) > 3:
                _examples += f" +{len(_judge_tasks) - 3} more"
            console.print(
                f"\n[yellow]{len(_judge_tasks)} task(s) need an LLM judge and will be skipped[/yellow] "
                f"[dim]({_examples})[/dim]"
            )
            _ans = console.input("[bold]Enable LLM judge now? [y/N] [/bold]").strip().lower()
            if _ans == "y":
                _default = all_pairs[0][0].id
                _model_input = console.input(f"  Judge model [blank = {_default!r}]: ").strip()
                judge_model = _model_input or _default
                _discovered_ids = {m.id for m, _ in all_pairs}
                if judge_model in _discovered_ids:
                    judge_client = next(b for m, b in all_pairs if m.id == judge_model).get_openai_client()
                else:
                    # External model (e.g. OpenAI, Anthropic-compat) — collect credentials.
                    _base_url = console.input("  API base URL (e.g. https://api.openai.com/v1): ").strip()
                    _env_key = os.environ.get("OPENAI_API_KEY", "")
                    _key_hint = "set via $OPENAI_API_KEY" if _env_key else "required"
                    _api_key = console.input(f"  API key [{_key_hint}]: ").strip() or _env_key
                    from openai import OpenAI as _OpenAI
                    judge_client = _OpenAI(api_key=_api_key or "noop", base_url=_base_url or None)
                console.print(f"  [green]✓[/green] Judge enabled — model: [bold]{judge_model}[/bold]\n")

    cats = sorted(set(t["category"] for t in tasks))
    console.print(
        f"[dim]{len(tasks)} tasks · categories: {cats} · "
        f"{len(all_pairs)} model(s)[/dim]\n"
    )

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
        )
        print_arena_leaderboard(players)
        save_arena_results(players, args.output)
        return

    all_results: dict = {}

    # Incremental JSONL for crash safety (one line per task, written immediately)
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _out_dir = Path(args.output)
    _out_dir.mkdir(exist_ok=True)
    _resume = bench_config.get("resume", False) or args.resume
    _cached_records: dict = {}
    if _resume:
        _existing = sorted(_out_dir.glob("results_*.jsonl"))
        if _existing:
            _jsonl_path = _existing[-1]
            with _jsonl_path.open(encoding="utf-8") as _fh:
                for _line in _fh:
                    try:
                        _rec = json.loads(_line)
                        _key = _record_cache_key(_rec)
                        if _key[0] and _key[1] and _key[3]:
                            _cached_records[_key] = _rec
                    except (ValueError, KeyError):
                        pass
            console.print(f"[dim]Resuming from {_jsonl_path.name} — {len(_cached_records)} compatible cached result(s)[/dim]")
        else:
            _jsonl_path = _out_dir / f"results_{_ts}.jsonl"
            console.print("[dim]--resume: no existing JSONL found — starting fresh[/dim]")
    else:
        _jsonl_path = _out_dir / f"results_{_ts}.jsonl"
    console.print(f"[dim]Streaming results → {_jsonl_path}[/dim]\n")

    for model_info, backend in all_pairs:
        label = f"{model_info.id}  [dim]({backend.name})[/dim]"
        console.rule(f"[bold cyan]{label}[/bold cyan]")

        model_results = _run_model(
            model_info=model_info,
            backend=backend,
            tasks=tasks,
            bench_config=bench_config,
            cached_records=_cached_records,
            jsonl_path=_jsonl_path,
            allow_code_exec=args.allow_code_exec,
            no_autoload=args.no_autoload,
            judge_client=judge_client,
            judge_model=judge_model,
        )
        all_results[model_info.id] = model_results

        passed   = sum(1 for r in model_results if r["score"] >= 1.0)
        tps_vals = [r["tps"] for r in model_results if r.get("tps")]
        avg_tps  = sum(tps_vals) / len(tps_vals) if tps_vals else 0
        score_pct = passed / len(model_results) * 100 if model_results else 0
        console.print(
            f"\n  [bold]Score: {passed}/{len(model_results)} "
            f"({score_pct:.0f}%)  "
            f"Avg TPS: {avg_tps:.1f}[/bold]\n"
        )

    print_report(all_results)
    save_results(all_results, args.output)
    if args.html_report:
        save_html_report(all_results, args.output)

    # ── CI threshold check ────────────────────────────────────────────────
    if args.ci_threshold is not None:
        all_scored = [r for rs in all_results.values() for r in rs]
        if all_scored:
            ratio = sum(r["score"] for r in all_scored) / len(all_scored)
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
