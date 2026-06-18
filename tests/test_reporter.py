"""
tests/test_reporter.py — unit tests for reporter helper functions.

Run with:  pytest tests/test_reporter.py -v
"""
from io import StringIO

import pytest

import benchmark.reporter as reporter
from benchmark.console import make_console
from benchmark.evaluation import CATEGORY_WEIGHTS
from benchmark.reporter import _composite_score, _coverage_counts


def make_result(category: str, score: float, **task_extra) -> dict:
    return {
        "task": {"id": "t", "category": category, **task_extra},
        "score": score,
        "tps": None,
        "ttft_ms": None,
        "total_ms": None,
    }


class TestCompositeScore:
    def test_empty_returns_none(self):
        assert _composite_score([]) is None

    def test_single_category_average(self):
        results = [make_result("math", 1.0), make_result("math", 0.0)]
        c = _composite_score(results)
        assert c == pytest.approx(0.5)

    def test_weights_applied(self):
        """coding (1.5) at 100% vs writing (0.8) at 0% => composite > 0.5."""
        results = [make_result("coding", 1.0), make_result("writing", 0.0)]
        c = _composite_score(results)
        expected = (1.0 * CATEGORY_WEIGHTS["coding"] + 0.0 * CATEGORY_WEIGHTS["writing"]) / (
            CATEGORY_WEIGHTS["coding"] + CATEGORY_WEIGHTS["writing"]
        )
        assert c == pytest.approx(expected)
        assert c > 0.5  # coding outweighs writing

    def test_unknown_category_defaults_weight_1(self):
        results = [make_result("novel_category", 1.0)]
        c = _composite_score(results)
        assert c == pytest.approx(1.0)

    def test_all_perfect(self):
        results = [make_result(cat, 1.0) for cat in CATEGORY_WEIGHTS]
        c = _composite_score(results)
        assert c == pytest.approx(1.0)

    def test_all_zero(self):
        results = [make_result(cat, 0.0) for cat in CATEGORY_WEIGHTS]
        c = _composite_score(results)
        assert c == pytest.approx(0.0)

    def test_multiple_tasks_per_category(self):
        """Average within each category before weighting."""
        results = [
            make_result("math", 1.0),
            make_result("math", 0.0),  # avg = 0.5
            make_result("coding", 1.0),  # avg = 1.0
        ]
        c = _composite_score(results)
        expected = (0.5 * CATEGORY_WEIGHTS["math"] + 1.0 * CATEGORY_WEIGHTS["coding"]) / (
            CATEGORY_WEIGHTS["math"] + CATEGORY_WEIGHTS["coding"]
        )
        assert c == pytest.approx(expected)

    def test_default_composite_excludes_smoke_diagnostic_and_high_contamination(self):
        results = [
            make_result("coding", 1.0, benchmark_tier="leaderboard"),
            make_result("coding", 0.0, benchmark_tier="smoke"),
            make_result("coding", 0.0, benchmark_tier="diagnostic"),
            make_result("coding", 0.0, contamination_risk="high"),
        ]

        assert _composite_score(results) == pytest.approx(1.0)
        assert _composite_score(results, core_only=False) == pytest.approx(0.25)

    def test_coverage_counts_expected_task_matrix(self):
        expected = [
            {"id": "task_a", "category": "coding"},
            {"id": "task_b", "category": "coding"},
        ]
        results = [make_result("coding", 1.0, id="task_a")]

        assert _coverage_counts(results, expected) == (1, 2)

    def test_composite_is_suppressed_for_incomplete_expected_task_matrix(self):
        expected = [
            {"id": "task_a", "category": "coding"},
            {"id": "task_b", "category": "coding"},
        ]
        results = [make_result("coding", 1.0, id="task_a")]

        assert _composite_score(results, expected_tasks=expected) is None


def test_print_task_result_shows_agent_loop_progress(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(reporter, "console", make_console(file=stream))
    result = make_result("agent_loop", 0.0, id="agent_loop_partial")
    result.update({
        "score_detail": "agent_loop: max steps reached without final",
        "agent_loop_progress_passed": 3,
        "agent_loop_progress_total": 7,
        "agent_loop_termination": "max_steps",
    })

    reporter.print_task_result(result)

    output = stream.getvalue()
    assert "progress=3/7" in output
    assert "termination=max_steps" in output
