"""
tests/test_scorer.py — unit tests for all 15 scoring types.

Run with:  pytest tests/test_scorer.py -v
"""
import pytest
from benchmark.scorer import score_response, score_pass_at_k


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

    def test_accent_insensitive_needle(self):
        # Task stores unaccented needle; model response contains accented character
        s, _ = score({"type": "contains", "value": "brasilia"}, "The capital is Brasília.")
        assert s == 1.0

    def test_accent_insensitive_response(self):
        # Needle contains accent; response is unaccented
        s, _ = score({"type": "contains", "value": "Brasília"}, "The capital is Brasilia.")
        assert s == 1.0


# ── contains_n ────────────────────────────────────────────────────────────────

class TestContainsN:
    def test_meets_min_exactly(self):
        s, _ = score({"type": "contains_n", "answer": "network", "min_count": 3},
                     "The network connects to another network via a third network.")
        assert s == 1.0

    def test_exceeds_min(self):
        s, _ = score({"type": "contains_n", "answer": "cat", "min_count": 2},
                     "A cat sat near another cat and yet another cat.")
        assert s == 1.0

    def test_below_min(self):
        s, d = score({"type": "contains_n", "answer": "network", "min_count": 3},
                     "This is a network.")
        assert s == 0.0
        assert "1" in d

    def test_zero_occurrences(self):
        s, _ = score({"type": "contains_n", "answer": "quantum", "min_count": 1},
                     "Nothing relevant here.")
        assert s == 0.0

    def test_default_min_count_one(self):
        s, _ = score({"type": "contains_n", "answer": "hello"}, "Just say hello.")
        assert s == 1.0

    def test_accent_insensitive(self):
        # Needle is unaccented; response contains accented form repeated 2x
        s, _ = score({"type": "contains_n", "answer": "cafe", "min_count": 2},
                     "Visit the café on Main St. The café opens at 8.")
        assert s == 1.0


# ── not_contains ──────────────────────────────────────────────────────────────

class TestNotContains:
    def test_all_absent(self):
        s, _ = score({"type": "not_contains", "forbidden": ["sorry", "cannot"]},
                     "The boiling point is 100.")
        assert s == 1.0

    def test_one_present(self):
        s, d = score({"type": "not_contains", "forbidden": ["sorry", "cannot"]},
                     "I cannot provide that information.")
        assert s == 0.0
        assert "cannot" in d

    def test_multiple_present(self):
        s, d = score({"type": "not_contains", "forbidden": ["pet", "friend", "loyal"]},
                     "Dogs are loyal pets and friendly companions.")
        assert s == 0.0
        assert len([f for f in ["pet", "loyal"] if f in d]) > 0

    def test_case_insensitive(self):
        s, d = score({"type": "not_contains", "forbidden": ["sorry"]},
                     "SORRY, I can't help.")
        assert s == 0.0

    def test_empty_forbidden_list(self):
        s, _ = score({"type": "not_contains", "forbidden": []}, "Any response here.")
        assert s == 1.0


# ── ends_with ─────────────────────────────────────────────────────────────────

class TestEndsWith:
    def test_last_word_matches(self):
        s, _ = score({"type": "ends_with", "answer": "ocean"},
                     "Life on Earth depends on the ocean.")
        assert s == 1.0

    def test_last_word_wrong(self):
        s, d = score({"type": "ends_with", "answer": "ocean"},
                     "Life on Earth depends on the sea.")
        assert s == 0.0
        assert "sea" in d

    def test_trailing_punctuation_stripped(self):
        s, _ = score({"type": "ends_with", "answer": "ocean"},
                     "The most important body of water is the ocean!")
        assert s == 1.0

    def test_multiline_uses_last_line(self):
        s, _ = score({"type": "ends_with", "answer": "ocean"},
                     "Some intro text.\nThe key resource is the ocean.")
        assert s == 1.0

    def test_empty_response(self):
        s, d = score({"type": "ends_with", "answer": "ocean"}, "")
        assert s == 0.0
        assert "Empty" in d

    def test_accent_insensitive(self):
        # Task expects unaccented 'resume'; model ends with accented 'résumé'
        s, _ = score({"type": "ends_with", "answer": "resume"},
                     "Please attach your résumé.")
        assert s == 1.0


# ── word_count ────────────────────────────────────────────────────────────────

class TestWordCount:
    def test_within_range(self):
        text = " ".join(["word"] * 60)
        s, d = score({"type": "word_count", "min": 50, "max": 70}, text)
        assert s == 1.0
        assert "60" in d

    def test_below_min(self):
        text = " ".join(["word"] * 30)
        s, _ = score({"type": "word_count", "min": 50, "max": 70}, text)
        assert s == 0.0

    def test_above_max(self):
        text = " ".join(["word"] * 100)
        s, _ = score({"type": "word_count", "min": 50, "max": 70}, text)
        assert s == 0.0

    def test_exactly_at_min(self):
        text = " ".join(["word"] * 50)
        s, _ = score({"type": "word_count", "min": 50, "max": 70}, text)
        assert s == 1.0

    def test_exactly_at_max(self):
        text = " ".join(["word"] * 70)
        s, _ = score({"type": "word_count", "min": 50, "max": 70}, text)
        assert s == 1.0

    def test_only_min_specified(self):
        text = " ".join(["word"] * 10)
        s, _ = score({"type": "word_count", "min": 5}, text)
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

    def test_value_key(self):
        """Tasks use 'value' key (e.g. writing.yaml); scorer must accept it."""
        s, _ = score({"type": "line_count", "value": 3}, "Line 1\nLine 2\nLine 3")
        assert s == 1.0

    def test_too_few(self):
        s, d = score({"type": "line_count", "count": 5}, "Only\nTwo")
        assert s == 0.0
        assert "2" in d

    def test_blank_lines_ignored(self):
        s, _ = score({"type": "line_count", "count": 2}, "Line 1\n\nLine 2\n\n")
        assert s == 1.0

    def test_missing_key_returns_error(self):
        s, d = score({"type": "line_count"}, "Line 1\nLine 2")
        assert s == 0.0
        assert "value" in d.lower() or "count" in d.lower()


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


# ── llm_judge ─────────────────────────────────────────────────────────────────

class TestLLMJudge:
    def test_no_client_returns_skip_detail(self):
        """Without a judge client, llm_judge score is 0 with informative detail."""
        s, d = score(
            {"type": "llm_judge", "criteria": "Is the answer helpful?"},
            "The answer is 42.",
        )
        assert s == 0.0  # None coerced to 0.0
        assert "judge" in d.lower()

    def test_judge_parses_score_line(self):
        """Mock client returns SCORE: 8 — normalised to 0.8."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "The response is largely correct.\nSCORE: 8"
        )
        s, d = score(
            {"type": "llm_judge", "criteria": "Accuracy"},
            "My answer.",
            judge_client=mock_client,
            judge_model="mock-judge",
        )
        assert s == pytest.approx(0.8)
        assert "8/10" in d

    def test_judge_score_10_capped_at_1(self):
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "SCORE: 10"
        s, _ = score(
            {"type": "llm_judge", "criteria": "Anything"},
            "Perfect answer.",
            judge_client=mock_client,
            judge_model="mock-judge",
        )
        assert s == 1.0

    def test_judge_handles_unparseable_response(self):
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "I cannot determine a score."
        )
        s, d = score(
            {"type": "llm_judge", "criteria": "Anything"},
            "Some response.",
            judge_client=mock_client,
            judge_model="mock-judge",
        )
        assert s == 0.0
        assert "could not parse" in d.lower()

    def test_judge_client_exception_handled(self):
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        s, d = score(
            {"type": "llm_judge", "criteria": "Anything"},
            "Some response.",
            judge_client=mock_client,
            judge_model="mock-judge",
        )
        assert s == 0.0
        assert "error" in d.lower()

    def test_judge_score_in_middle_of_response(self):
        """SCORE: N on any line (searched in reverse) is accepted."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "Step 1: check accuracy.\nSCORE: 7\nOverall assessment done."
        )
        s, d = score(
            {"type": "llm_judge", "criteria": "Accuracy"},
            "Answer.",
            judge_client=mock_client,
            judge_model="mock-judge",
        )
        assert s == pytest.approx(0.7)
        assert "7/10" in d


# ── fuzzy_match (accent) ──────────────────────────────────────────────────────

class TestFuzzyMatch:
    def test_answer_in_response(self):
        s, _ = score({"type": "fuzzy_match", "value": "artificial intelligence"},
                     "AI stands for artificial intelligence in computing.")
        assert s == 1.0

    def test_response_in_answer(self):
        s, _ = score({"type": "fuzzy_match", "value": "Paris is the capital of France"},
                     "Paris")
        assert s == 1.0

    def test_no_match(self):
        s, _ = score({"type": "fuzzy_match", "value": "quantum mechanics"},
                     "The weather is sunny today.")
        assert s == 0.0

    def test_accent_insensitive(self):
        s, _ = score({"type": "fuzzy_match", "value": "naive"},
                     "A naïve approach was taken.")
        assert s == 1.0


# ── pass_at_k ─────────────────────────────────────────────────────────────────

class TestPassAtK:
    def _make_raw(self, response: str) -> dict:
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

    def _pass_k_task(self, inner_type: str = "contains", inner_scoring: dict | None = None):
        base = {"type": "pass_at_k", "inner_type": inner_type, "k": 3}
        if inner_scoring:
            base.update(inner_scoring)
        return {
            "id": "pk_task",
            "prompt": "test",
            "category": "coding",
            "scoring": base,
        }

    def test_all_pass_gives_1(self):
        task = self._pass_k_task("contains", {"value": "hello"})
        runs = [self._make_raw("hello world")] * 3
        result = score_pass_at_k(task, runs)
        assert result["score"] == pytest.approx(1.0)

    def test_none_pass_gives_0(self):
        task = self._pass_k_task("contains", {"value": "hello"})
        runs = [self._make_raw("goodbye world")] * 3
        result = score_pass_at_k(task, runs)
        assert result["score"] == pytest.approx(0.0)

    def test_partial_pass_between_0_and_1(self):
        task = self._pass_k_task("contains", {"value": "hello"})
        runs = [self._make_raw("hello"), self._make_raw("nope"), self._make_raw("nope")]
        result = score_pass_at_k(task, runs)
        # 1 out of 3 passes; pass@3 with n=3, c=1 => 1 - C(2,3)/C(3,3) = 1 - 0 = 1.0
        # Actually: C(n-c, k) / C(n, k) = C(2,3)/C(3,3); C(2,3)=0, so estimate=1.0
        # For n=k=3, c=1: any-pass = True, so estimate=1.0
        assert result["score"] > 0.0

    def test_score_detail_contains_pass_fraction(self):
        task = self._pass_k_task("contains", {"value": "hello"})
        runs = [self._make_raw("hello")] * 2 + [self._make_raw("nope")]
        result = score_pass_at_k(task, runs)
        assert "pass@" in result["score_detail"]
        assert "/3" in result["score_detail"]

    def test_n_greater_than_k_uses_estimator(self):
        """When n > k, the Chen et al. estimator is used."""
        task = self._pass_k_task("contains", {"value": "hello"})
        task["scoring"]["k"] = 2  # k=2 but we pass n=5 samples
        # 3 of 5 pass: pass@2 = 1 - C(2,2)/C(5,2) = 1 - 1/10 = 0.9
        runs = (
            [self._make_raw("hello")] * 3
            + [self._make_raw("nope")] * 2
        )
        result = score_pass_at_k(task, runs)
        assert result["score"] == pytest.approx(0.9, abs=1e-9)


# ── logprob_choice ────────────────────────────────────────────────────────────

class TestLogprobChoice:
    def test_exact_match(self):
        s, d = score({"type": "logprob_choice", "value": "B"}, "B")
        assert s == 1.0
        assert "Logprob match" in d

    def test_case_insensitive(self):
        s, _ = score({"type": "logprob_choice", "value": "C"}, "c")
        assert s == 1.0

    def test_mismatch(self):
        s, d = score({"type": "logprob_choice", "value": "A"}, "D")
        assert s == 0.0
        assert "expected 'A'" in d

    def test_legacy_answer_key(self):
        s, _ = score({"type": "logprob_choice", "answer": "B"}, "B")
        assert s == 1.0

    def test_whitespace_stripped(self):
        s, _ = score({"type": "logprob_choice", "value": "A"}, "  A  ")
        assert s == 1.0

