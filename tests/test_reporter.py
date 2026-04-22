"""
tests/test_reporter.py — unit tests for reporter helper functions.

Run with:  pytest tests/test_reporter.py -v
"""
import pytest
from benchmark.reporter import CATEGORY_WEIGHTS, _composite_score


def make_result(category: str, score: float) -> dict:
    return {
        "task": {"id": "t", "category": category},
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
