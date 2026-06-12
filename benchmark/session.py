"""
benchmark/session.py — orchestration helpers for a benchmark run.

Everything the CLI needs between "parse args" and "print the report":
config loading, backend/model discovery, judge resolution, resume cache,
and the per-model task loop. Kept separate from benchmark/cli.py so the
pieces are importable and testable without argparse.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import yaml
from rich.table import Table
from rich import box

from benchmark.backends import create_backend
from benchmark.backends.base import ModelInfo
from benchmark.console import make_console
from benchmark.evaluation import annotate_pass
from benchmark.reporter import print_task_result, append_jsonl
from benchmark.result import cache_key, record_cache_key, from_record
from benchmark.runner import ModelRunner
from benchmark.scorer import score_response, score_pass_at_k

console = make_console()


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        console.print(f"[red]Config not found: {cfg_path}[/red]")
        sys.exit(1)
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def discover_models(config: dict, backend_filter: str | None = None) -> list[tuple[ModelInfo, object]]:
    """
    Returns list of (ModelInfo, Backend instance) pairs for all enabled backends.
    backend_filter: restrict to a specific backend type name.
    """
    pairs: list[tuple[ModelInfo, object]] = []
    backends_cfg: dict = config.get("backends", {})
    manual_ids: list[str] = config.get("models", [])
    reachable_backends: list = []

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

        for m in discovered:
            pairs.append((m, backend))
        reachable_backends.append(backend)

    # Manual models that no backend discovered: synthesize once, on the first
    # reachable backend — one phantom per remaining backend would create
    # duplicate pairs that overwrite real runs in all_results (keyed by id).
    if manual_ids and reachable_backends:
        found_ids = {m.id for m, _ in pairs}
        first = reachable_backends[0]
        for mid in manual_ids:
            if mid not in found_ids:
                pairs.append((ModelInfo(id=mid, name=mid, backend_name=first.name), first))

    # If no backends are enabled, fall back to manual list with no backend
    if not pairs and manual_ids and not backend_filter:
        console.print("[yellow]No enabled backends are reachable. Manual model list requires a reachable backend.[/yellow]")

    return pairs


def print_discovery_table(pairs: list[tuple[ModelInfo, object]]) -> None:
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


def build_judge(args, config: dict, all_pairs: list, tasks: list[dict]) -> tuple[object | None, str | None]:
    """
    Resolve (judge_client, judge_model) — precedence: --judge-model CLI flag,
    then config.yaml's judge block, then an interactive TTY prompt offered
    only when the selected tasks actually need a judge.
    """
    judge_cfg = config.get("judge", {})
    judge_client = None
    judge_model: str | None = None

    if args.judge_model:
        # CLI flag takes precedence over config and the interactive prompt.
        discovered_ids = {m.id for m, _ in all_pairs}
        if args.judge_model in discovered_ids:
            judge_model = args.judge_model
            judge_client = next(b for m, b in all_pairs if m.id == judge_model).get_openai_client()
        else:
            # External model: base URL required; key from flag or env.
            base_url = args.judge_base_url
            if not base_url:
                console.print(
                    f"[red]--judge-model {args.judge_model!r} is not a discovered local model. "
                    "Provide --judge-base-url for an external API endpoint.[/red]"
                )
                sys.exit(1)
            api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY", "noop")
            from openai import OpenAI as _OpenAI
            judge_client = _OpenAI(api_key=api_key, base_url=base_url)
            judge_model = args.judge_model
        console.print(f"[dim]LLM judge enabled (--judge-model) — model: {judge_model}[/dim]")
        return judge_client, judge_model

    if judge_cfg.get("enabled") and all_pairs:
        judge_model = judge_cfg.get("model") or all_pairs[0][0].id
        # Use the backend that actually serves the judge model (mirrors the
        # --judge-model path); fall back to the first pair if undiscovered.
        judge_backend = next((b for m, b in all_pairs if m.id == judge_model), all_pairs[0][1])
        judge_client = judge_backend.get_openai_client()
        console.print(f"[dim]LLM judge enabled — model: {judge_model}[/dim]")
        return judge_client, judge_model

    # Interactive prompt: only when judge tasks would otherwise be skipped and
    # we're running interactively (not piped / CI).
    if sys.stdin.isatty():
        judge_types = {"llm_judge", "rubric_judge"}
        judge_tasks = [t for t in tasks if t.get("scoring", {}).get("type") in judge_types]
        if judge_tasks:
            examples = ", ".join(t["id"] for t in judge_tasks[:3])
            if len(judge_tasks) > 3:
                examples += f" +{len(judge_tasks) - 3} more"
            console.print(
                f"\n[yellow]{len(judge_tasks)} task(s) need an LLM judge and will be skipped[/yellow] "
                f"[dim]({examples})[/dim]"
            )
            ans = console.input("[bold]Enable LLM judge now? [y/N] [/bold]").strip().lower()
            if ans == "y":
                default = all_pairs[0][0].id
                model_input = console.input(f"  Judge model [blank = {default!r}]: ").strip()
                judge_model = model_input or default
                discovered_ids = {m.id for m, _ in all_pairs}
                if judge_model in discovered_ids:
                    judge_client = next(b for m, b in all_pairs if m.id == judge_model).get_openai_client()
                else:
                    # External model (e.g. OpenAI, Anthropic-compat) — collect credentials.
                    base_url = console.input("  API base URL (e.g. https://api.openai.com/v1): ").strip()
                    env_key = os.environ.get("OPENAI_API_KEY", "")
                    key_hint = "set via $OPENAI_API_KEY" if env_key else "required"
                    api_key = console.input(f"  API key [{key_hint}]: ").strip() or env_key
                    from openai import OpenAI as _OpenAI
                    judge_client = _OpenAI(api_key=api_key or "noop", base_url=base_url or None)
                console.print(f"  [green]✓[/green] Judge enabled — model: [bold]{judge_model}[/bold]\n")

    return judge_client, judge_model


def load_cached_records(out_dir: Path) -> tuple[dict, Path | None]:
    """
    Load the most recent results_*.jsonl in out_dir into a {cache_key: record}
    dict for --resume. Returns ({}, None) when there is nothing to resume.
    """
    existing = sorted(out_dir.glob("results_*.jsonl"))
    if not existing:
        return {}, None
    jsonl_path = existing[-1]
    cached: dict = {}
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                key = record_cache_key(rec)
                if key[0] and key[1] and key[3]:
                    cached[key] = rec
            except (ValueError, KeyError):
                pass
    return cached, jsonl_path


def run_model(
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
        key = cache_key(model_info.id, task)
        if key in cached_records:
            console.print(f"  [dim]↩  {task['id']:40s}  (cached — skipped)[/dim]")
            model_results.append(from_record(task, cached_records[key]))
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
                    annotate_pass(scored)

        scored["model_id"] = model_info.id
        model_results.append(scored)
        print_task_result(scored)
        append_jsonl(scored, jsonl_path)

    return model_results
