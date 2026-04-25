"""
benchmark/arena.py — Subjective ELO Arena (Chatbot Arena-style pairwise comparison).

Pits two models against each other on the same prompt.  An LLM judge
declares a winner for each round, and ELO ratings are updated accordingly.

Usage (from run.py):
    python run.py --arena

How it works:
  1. For each task, send the same prompt to Model A and Model B.
  2. A judge LLM reads both responses (order randomised to reduce position bias).
  3. The judge outputs WINNER: A | B | TIE.
  4. ELO ratings are updated using the standard chess formula.
  5. After all rounds, a leaderboard table is printed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich import box

from benchmark.runner import ModelRunner
from benchmark.utils import strip_thinking

console = Console()

# ── ELO engine ───────────────────────────────────────────────────────────────

_INITIAL_ELO = 1500
_K_FACTOR = 32


@dataclass
class ArenaPlayer:
    model_id: str
    backend_name: str
    elo: float = _INITIAL_ELO
    wins: int = 0
    losses: int = 0
    ties: int = 0
    history: list[dict] = field(default_factory=list)


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def _update_elo(player_a: ArenaPlayer, player_b: ArenaPlayer, result: str):
    """
    Update ELO ratings for both players based on a match result.
    result: 'A', 'B', or 'TIE'
    """
    ea = _expected(player_a.elo, player_b.elo)
    eb = _expected(player_b.elo, player_a.elo)

    if result == "A":
        sa, sb = 1.0, 0.0
        player_a.wins += 1
        player_b.losses += 1
    elif result == "B":
        sa, sb = 0.0, 1.0
        player_a.losses += 1
        player_b.wins += 1
    else:  # TIE
        sa, sb = 0.5, 0.5
        player_a.ties += 1
        player_b.ties += 1

    player_a.elo += _K_FACTOR * (sa - ea)
    player_b.elo += _K_FACTOR * (sb - eb)


# ── Judge protocol ───────────────────────────────────────────────────────────

_ARENA_JUDGE_SYSTEM = (
    "You are a fair and impartial judge comparing two AI assistant responses.\n"
    "You will see a question and two responses labeled [A] and [B].\n"
    "Evaluate both on: accuracy, helpfulness, clarity, and depth.\n"
    "Think step by step, then on your LAST LINE output exactly one of:\n"
    "  WINNER: A\n"
    "  WINNER: B\n"
    "  WINNER: TIE\n"
    "Do NOT explain after the WINNER line."
)


def _judge_pair(
    prompt: str,
    response_a: str,
    response_b: str,
    judge_client,
    judge_model: str | None,
) -> tuple[str, str]:
    """
    Ask the judge to compare two responses.
    Returns (result, judge_reasoning) where result is 'A', 'B', or 'TIE'.
    The display order is randomised, and the mapping is reversed in the result
    so position bias does not systematically favour one model.
    """
    import re

    # Randomise display order to combat position bias
    swap = random.random() < 0.5
    if swap:
        displayed_a, displayed_b = response_b, response_a
    else:
        displayed_a, displayed_b = response_a, response_b

    user_msg = (
        f"Question:\n<question>\n{prompt}\n</question>\n\n"
        f"Response [A]:\n<response_a>\n{displayed_a[:4000]}\n</response_a>\n\n"
        f"Response [B]:\n<response_b>\n{displayed_b[:4000]}\n</response_b>"
    )

    try:
        resp = judge_client.chat.completions.create(
            model=judge_model or "default",
            messages=[
                {"role": "system", "content": _ARENA_JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
            timeout=90,
        )
        judge_text = resp.choices[0].message.content or ""
    except Exception as exc:
        return "TIE", f"Judge error: {exc}"

    # Parse WINNER from last line
    lines = [ln.strip() for ln in judge_text.strip().split("\n") if ln.strip()]
    result = "TIE"
    for ln in reversed(lines):
        m = re.search(r"WINNER:\s*(A|B|TIE)", ln, re.IGNORECASE)
        if m:
            raw = m.group(1).upper()
            if swap:
                # Un-swap: if judge said A wins but we swapped, real winner is B
                raw = {"A": "B", "B": "A"}.get(raw, raw)
            result = raw
            break

    return result, judge_text


# ── Arena runner ─────────────────────────────────────────────────────────────

def run_arena(
    model_pairs: list[tuple],  # list of (ModelInfo, Backend)
    tasks: list[dict],
    bench_config: dict,
    judge_client,
    judge_model: str | None,
) -> dict[str, ArenaPlayer]:
    """
    Run a round-robin arena: every pair of models competes on every task.
    Returns a dict of model_id -> ArenaPlayer with final ELO ratings.
    """
    if len(model_pairs) < 2:
        console.print("[red]Arena mode requires at least 2 models.[/red]")
        return {}

    # Create players
    players: dict[str, ArenaPlayer] = {}
    runners: dict[str, ModelRunner] = {}
    for model_info, backend in model_pairs:
        mid = model_info.id
        players[mid] = ArenaPlayer(model_id=mid, backend_name=backend.name)
        runners[mid] = ModelRunner(backend, mid, bench_config)

    model_ids = list(players.keys())
    total_matchups = len(tasks) * len(model_ids) * (len(model_ids) - 1) // 2
    completed = 0

    console.print(f"\n[bold]Arena Mode[/bold]  [dim]{len(model_ids)} models × {len(tasks)} tasks = {total_matchups} matchups[/dim]\n")

    for task in tasks:
        # Pre-generate all responses for this task
        responses: dict[str, str] = {}
        for mid in model_ids:
            raw = runners[mid].run_task(task)
            responses[mid] = strip_thinking(raw.get("response", ""))

        # Round-robin pairwise comparison
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                mid_a, mid_b = model_ids[i], model_ids[j]
                result, reasoning = _judge_pair(
                    prompt=task["prompt"],
                    response_a=responses[mid_a],
                    response_b=responses[mid_b],
                    judge_client=judge_client,
                    judge_model=judge_model,
                )
                _update_elo(players[mid_a], players[mid_b], result)

                completed += 1
                winner_label = mid_a if result == "A" else (mid_b if result == "B" else "TIE")

                # Record match history
                match_record = {
                    "task_id": task["id"],
                    "opponent": mid_b,
                    "result": result,
                }
                players[mid_a].history.append(match_record)
                players[mid_b].history.append({**match_record, "opponent": mid_a,
                                                "result": {"A": "B", "B": "A"}.get(result, result)})

                icon = {"A": "🏆", "B": "🏆", "TIE": "🤝"}.get(result, "?")
                console.print(
                    f"  {icon}  [dim]{task['id']:30s}[/dim]  "
                    f"{mid_a[:25]:25s} vs {mid_b[:25]:25s}  "
                    f"→ [bold]{winner_label[:30]}[/bold]  "
                    f"[dim]({completed}/{total_matchups})[/dim]"
                )

    return players


def print_arena_leaderboard(players: dict[str, ArenaPlayer]):
    """Print a sorted ELO leaderboard."""
    if not players:
        return

    console.print("\n")
    console.rule("[bold white]ARENA LEADERBOARD[/bold white]")

    t = Table(box=box.ROUNDED, title="ELO Ratings", show_lines=True)
    t.add_column("Rank", style="bold", justify="center", width=6)
    t.add_column("Model", style="cyan", min_width=30)
    t.add_column("Backend", style="dim", min_width=12)
    t.add_column("ELO", style="bold", justify="center", min_width=8)
    t.add_column("W", style="green", justify="center", width=5)
    t.add_column("L", style="red", justify="center", width=5)
    t.add_column("T", style="yellow", justify="center", width=5)
    t.add_column("Win Rate", justify="center", min_width=10)

    sorted_players = sorted(players.values(), key=lambda p: p.elo, reverse=True)
    for rank, p in enumerate(sorted_players, 1):
        total = p.wins + p.losses + p.ties
        win_rate = f"{p.wins / total * 100:.0f}%" if total > 0 else "—"
        t.add_row(
            str(rank),
            p.model_id,
            p.backend_name,
            f"{p.elo:.0f}",
            str(p.wins),
            str(p.losses),
            str(p.ties),
            win_rate,
        )

    console.print(t)
