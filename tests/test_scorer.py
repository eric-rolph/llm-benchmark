"""
tests/test_scorer.py — unit tests for all 8 scoring methods.

Run with:  pytest tests/test_scorer.py -v
"""
import pytest
from benchmark.scorer import score_response


# ── helpers ──────────────────────────────────────────────────────────────────

def make_task(scoring: dict) -> dict:
    return {
        "id": "test_task",
        "prompt": "test",
        "category": "test",
        "scoring": scoring,
    }


def make_result(response: str) -> dict:
    return {
        "task_id": "test_task",
        "response": response,
        "error": None,
        "ttft_ms": None,
        "total_ms": 100.0,
        "tps": 10.0,
        "completion_tokens": 5,
        "reasoning_tokens": 0,
        "backend": "test",
    }


def score(task_scoring: dict, response: str, **kwargs) -> tuple:
    task = make_task(task_scoring)
    result = make_result(response)
    scored = score_response(task, result, **kwargs)
    return scored["score"], scored["score_detail"]


# ── numeric ───────────────────────────────────────────────────────────────────

class TestNumeric:
    def test_exact_integer(self):
        s, _ = score({"type": "numeric", "value": 42}, "The answer is 42.")
        assert s == 1.0

    def test_exact_float(self):
        s, _ = score({"type": "numeric", "value": 3.14, "tolerance": 0.01}, "Pi is approximately 3.14159")
        assert s == 1.0

    def test_within_tolerance(self):
        s, _ = score({"type": "numeric", "value": 100, "tolerance": 5}, "About 98.")
        assert s == 1.0

    def test_outside_tolerance(self):
        s, _ = score({"type": "numeric", "value": 100, "tolerance": 2}, "I think 95.")
        assert s == 0.0

    def test_no_number(self):
        s, d = score({"type": "numeric", "value": 7}, "I don't know.")
        assert s == 0.0
        assert "No number" in d

    def test_legacy_answer_key(self):
        """Backwards-compat: tasks using 'answer' instead of 'value'."""
        s, _ = score({"type": "numeric", "answer": 7}, "The answer is 7")
        assert s == 1.0

    def test_missing_value_raises_clear_error(self):
        """A task missing both 'answer' and 'value' should return score 0 with an error detail."""
        s, d = score({"type": "numeric"}, "42")
        assert s == 0.0
        assert "missing" in d.lower() or "No answer" in d


# ── exact ─────────────────────────────────────────────────────────────────────

class TestExact:
    def test_match(self):
        s, _ = score({"type": "exact", "value": "Paris"}, "Paris")
        assert s == 1.0

    def test_case_insensitive(self):
        s, _ = score({"type": "exact", "value": "paris"}, "PARIS")
        assert s == 1.0

    def test_no_match(self):
        s, _ = score({"type": "exact", "value": "Paris"}, "London")
        assert s == 0.0

    def test_legacy_answer_key(self):
        s, _ = score({"type": "exact", "answer": "Paris"}, "paris")
        assert s == 1.0


# ── contains ──────────────────────────────────────────────────────────────────

class TestContains:
    def test_found(self):
        s, _ = score({"type": "contains", "value": "photosynthesis"}, "Plants use photosynthesis to convert sunlight.")
        assert s == 1.0

    def test_not_found(self):
        s, _ = score({"type": "contains", "value": "mitosis"}, "Plants use photosynthesis.")
        assert s == 0.0

    def test_case_insensitive(self):
        s, _ = score({"type": "contains", "value": "PARIS"}, "The capital is paris")
        assert s == 1.0


# ── regex ─────────────────────────────────────────────────────────────────────

class TestRegex:
    def test_match(self):
        s, _ = score({"type": "regex", "pattern": r"\b\d{4}\b"}, "The year was 1969.")
        assert s == 1.0

    def test_no_match(self):
        s, _ = score({"type": "regex", "pattern": r"\b\d{4}\b"}, "Some text without years.")
        assert s == 0.0


# ── json_keys ─────────────────────────────────────────────────────────────────

class TestJsonKeys:
    def test_all_keys_present(self):
        s, _ = score(
            {"type": "json_keys", "keys": ["name", "age"]},
            'Here is the JSON: {"name": "Alice", "age": 30, "extra": true}',
        )
        assert s == 1.0

    def test_missing_key(self):
        s, d = score(
            {"type": "json_keys", "keys": ["name", "email"]},
            '{"name": "Alice"}',
        )
        assert s == 0.0
        assert "email" in d

    def test_no_json(self):
        s, d = score({"type": "json_keys", "keys": ["name"]}, "No JSON here")
        assert s == 0.0
        assert "No JSON" in d


# ── line_count ────────────────────────────────────────────────────────────────

class TestLineCount:
    def test_correct(self):
        s, _ = score({"type": "line_count", "count": 3}, "Line 1\nLine 2\nLine 3")
        assert s == 1.0

    def test_too_few(self):
        s, d = score({"type": "line_count", "count": 5}, "Only\nTwo")
        assert s == 0.0
        assert "2" in d

    def test_blank_lines_ignored(self):
        s, _ = score({"type": "line_count", "count": 2}, "Line 1\n\nLine 2\n\n")
        assert s == 1.0


# ── code_exec ─────────────────────────────────────────────────────────────────

class TestCodeExec:
    def test_disabled_by_default(self):
        """code_exec should be gated; disabled unless allow_code_exec=True."""
        s, d = score(
            {"type": "code_exec", "test_code": "print('PASS')"},
            "```python\ndef add(a, b): return a + b\n```",
        )
        assert s == 0.0
        assert "allow-code-exec" in d.lower() or "disabled" in d.lower()

    def test_enabled_passes(self):
        s, d = score(
            {"type": "code_exec", "test_code": "assert add(2, 3) == 5\nprint('PASS')"},
            "```python\ndef add(a, b):\n    return a + b\n```",
            allow_code_exec=True,
        )
        assert s == 1.0

    def test_enabled_fails(self):
        s, d = score(
            {"type": "code_exec", "test_code": "assert add(2, 3) == 5\nprint('PASS')"},
            "```python\ndef add(a, b):\n    return a - b  # wrong\n```",
            allow_code_exec=True,
        )
        assert s == 0.0


# ── error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_api_error_propagated(self):
        task = make_task({"type": "exact", "value": "Paris"})
        result = {**make_result(""), "error": "connection timeout", "response": ""}
        scored = score_response(task, result)
        assert scored["score"] == 0.0
        assert "connection timeout" in scored["score_detail"]

    def test_think_blocks_stripped(self):
        s, _ = score(
            {"type": "contains", "value": "Paris"},
            "<think>Let me think... The capital of France is Paris</think>Paris",
        )
        assert s == 1.0

    def test_unknown_scoring_type(self):
        s, d = score({"type": "nonexistent_type"}, "any response")
        assert s == 0.0
        assert "Unknown" in d
