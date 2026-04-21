"""
run.py — entry point for the LLM Benchmark Suite.

Supported backends (configure in config.yaml):
  lm_studio   — LM Studio local server  (http://localhost:1234)
  ollama      — Ollama                  (http://localhost:11434)
  llamacpp    — llama.cpp server        (http://localhost:8080)

Quick start:
  python run.py                          # auto-discover models, run all tasks
  python run.py --discover               # probe backends and list found models
  python run.py --dry-run                # validate task files + check backends, no inference
  python run.py --model "qwen3:8b"       # single model (all categories)
  python run.py --backend ollama         # only Ollama models
  python run.py --category math          # single category
  python run.py --task capital_france    # single task by ID
  python run.py --no-autoload            # skip LM Studio model-load attempt
  python run.py --allow-code-exec        # enable code_exec scoring (runs model-generated Python)
  python run.py --ci-threshold 0.8       # exit 1 if overall score < 80%  (CI integration)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table
from rich import box

from benchmark.backends import create_backend, discover_all_models
from benchmark.backends.base import ModelInfo
from benchmark.loader import load_tasks
from benchmark.reporter import print_report, print_task_result, save_results, append_jsonl
from benchmark.runner import ModelRunner
from benchmark.scorer import score_response

console = Console()

TASK_DIR = Path(__file__).parent / "tasks"


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
    parser.add_argument("--model",          help="Benchmark only this model ID (matched against any backend)")
    parser.add_argument("--backend",        help="Restrict to one backend type (lm_studio | ollama | llamacpp)")
    parser.add_argument("--category",       choices=categories,      help="Run only this task category")
    parser.add_argument("--task",           help="Run only a single task by ID")
    parser.add_argument("--output",         default="results",       help="Results output directory")
    parser.add_argument("--no-autoload",    action="store_true",     help="Skip LM Studio model-load attempt")
    parser.add_argument("--discover",       action="store_true",     help="Probe backends and print discovered models, then exit")
    parser.add_argument("--list-models",    action="store_true",     help="Alias for --discover")
    parser.add_argument("--dry-run",        action="store_true",     help="Validate task files and check backend connectivity, no inference")
    parser.add_argument("--allow-code-exec",action="store_true",     help="Enable code_exec scoring (runs model-generated Python locally — review tasks first)")
    parser.add_argument("--ci-threshold",   type=float, default=None,metavar="RATIO",
                        help="Exit with code 1 if overall score ratio is below this (e.g. 0.8 = 80%%)")
    args = parser.parse_args()

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
            tasks_all = load_tasks(validate=True)
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
        all_pairs = [(m, b) for m, b in all_pairs if args.model in m.id]
        if not all_pairs:
            console.print(f"[red]Model '{args.model}' not found in any enabled backend.[/red]")
            console.print("[dim]Run --discover to see available models.[/dim]")
            sys.exit(1)

    if not all_pairs:
        console.print("[red]No models to benchmark. Enable a backend in config.yaml or use --model.[/red]")
        console.print("[dim]Run --discover to see what is available.[/dim]")
        sys.exit(1)

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

    cats = sorted(set(t["category"] for t in tasks))
    console.print(
        f"[dim]{len(tasks)} tasks · categories: {cats} · "
        f"{len(all_pairs)} model(s)[/dim]\n"
    )

    all_results: dict = {}

    # Incremental JSONL for crash safety (one line per task, written immediately)
    from pathlib import Path as _Path
    import datetime as _dt
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _jsonl_path = _Path(args.output) / f"results_{_ts}.jsonl"
    _Path(args.output).mkdir(exist_ok=True)
    console.print(f"[dim]Streaming results → {_jsonl_path}[/dim]\n")

    for model_info, backend in all_pairs:
        label = f"{model_info.id}  [dim]({backend.name})[/dim]"
        console.rule(f"[bold cyan]{label}[/bold cyan]")

        runner = ModelRunner(backend, model_info.id, bench_config)

        auto_load = backend.config.get("auto_load", False)
        if auto_load and not args.no_autoload:
            runner.ensure_model_loaded()

        model_results: list = []
        for task in tasks:
            raw    = runner.run_task(task)
            scored = score_response(task, raw, allow_code_exec=args.allow_code_exec)
            model_results.append(scored)
            print_task_result(scored)
            append_jsonl(scored, _jsonl_path)

        all_results[model_info.id] = model_results

        passed   = sum(1 for r in model_results if r["score"] >= 1.0)
        tps_vals = [r["tps"] for r in model_results if r.get("tps")]
        avg_tps  = sum(tps_vals) / len(tps_vals) if tps_vals else 0
        console.print(
            f"\n  [bold]Score: {passed}/{len(model_results)} "
            f"({passed / len(model_results) * 100:.0f}%)  "
            f"Avg TPS: {avg_tps:.1f}[/bold]\n"
        )

    print_report(all_results)
    save_results(all_results, args.output)

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
