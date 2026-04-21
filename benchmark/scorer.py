"""
benchmark/scorer.py — scores model responses against task definitions.

Scoring types:
  numeric          — extract first number, compare with optional tolerance
  exact            — stripped case-insensitive equality
  contains         — substring check (case-insensitive)
  contains_n       — substring must appear at least N times (min_count)
  not_contains     — list of forbidden strings must all be absent
  ends_with        — last non-empty line's last word matches answer
  word_count       — word count must fall within [min, max]
  regex            — regex search
  json_keys        — parse JSON object, verify required keys exist
  line_count       — count non-empty lines, compare to expected
  code_exec        — extract code block, run it, look for PASS in stdout
  llm_judge        — placeholder (manual review or future LLM judge pass)
"""
import json
import os
import re
import subprocess
import sys
import tempfile

from benchmark.utils import strip_thinking


def score_response(task: dict, run_result: dict, allow_code_exec: bool = False) -> dict:
    scored = {
        **run_result,
        "task": task,
        "score": 0.0,
        "max_score": 1.0,
        "score_detail": "",
    }

    if run_result.get("error"):
        scored["score_detail"] = f"API error: {run_result['error']}"
        return scored

    # Strip any think/reasoning blocks that leaked into content
    response = strip_thinking(run_result["response"])
    scoring = task.get("scoring", {})
    method = scoring.get("type", "contains")

    _code_exec_fn = _score_code_exec if allow_code_exec else (
        lambda r, s: (0.0, "code_exec disabled — rerun with --allow-code-exec")
    )

    dispatch = {
        "numeric":      _score_numeric,
        "exact":        _score_exact,
        "contains":     _score_contains,
        "contains_n":   _score_contains_n,
        "not_contains": _score_not_contains,
        "ends_with":    _score_ends_with,
        "word_count":   _score_word_count,
        "regex":        _score_regex,
        "json_keys":    _score_json_keys,
        "line_count":   _score_line_count,
        "code_exec":    _code_exec_fn,
        "llm_judge":    lambda r, s: (None, "llm_judge (requires --judge flag)"),
    }

    fn = dispatch.get(method, lambda r, s: (0.0, f"Unknown scoring type: {method}"))
    score, detail = fn(response, scoring)
    scored["score"] = float(score) if score is not None else 0.0
    scored["score_detail"] = detail
    return scored


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_number(text: str) -> float | None:
    """Pull the first numeric value out of arbitrary text."""
    text = text.replace(",", "")
    for m in re.findall(r"-?\d+\.?\d*", text):
        try:
            return float(m)
        except ValueError:
            pass
    return None


def _extract_code(response: str) -> str:
    """Extract a Python code block from a model response."""
    # Prefer ```python ... ```
    m = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Any fenced block
    m = re.search(r"```\n?(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Heuristic: start from the first def / class / import line
    lines = response.strip().split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^(def |class |import |from )", line):
            return "\n".join(lines[i:]).strip()
    return response.strip()


# ── scorers ──────────────────────────────────────────────────────────────────

def _score_numeric(response: str, scoring: dict):
    # Accept both 'answer' (legacy) and 'value' keys
    raw = scoring.get("answer", scoring.get("value"))
    if raw is None:
        return 0.0, "No answer/value key in scoring definition"
    try:
        expected = float(raw)
    except (TypeError, ValueError):
        return 0.0, f"Invalid expected value: {raw!r}"
    tolerance = float(scoring.get("tolerance", 0))
    got = _extract_number(response)
    if got is None:
        return 0.0, f"No number found. Expected {expected}"
    if abs(got - expected) <= tolerance:
        return 1.0, f"Correct ({got})"
    return 0.0, f"Got {got}, expected {expected} ±{tolerance}"


def _score_exact(response: str, scoring: dict):
    raw = scoring.get("answer", scoring.get("value", ""))
    expected = str(raw).strip().lower()
    got = response.strip().lower()
    if got == expected:
        return 1.0, "Exact match"
    return 0.0, f"Got '{response.strip()}', expected '{raw}'"


def _score_contains(response: str, scoring: dict):
    raw = scoring.get("answer", scoring.get("value", ""))
    needle = str(raw).lower()
    if needle in response.lower():
        return 1.0, f"Contains '{raw}'"
    return 0.0, f"Missing '{raw}'"


def _score_contains_n(response: str, scoring: dict):
    needle = str(scoring.get("answer", scoring.get("value", ""))).lower()
    min_count = int(scoring.get("min_count", 1))
    count = response.lower().count(needle)
    if count >= min_count:
        return 1.0, f"'{needle}' appears {count}x (min {min_count})"
    return 0.0, f"'{needle}' appears {count}x, need at least {min_count}"


def _score_not_contains(response: str, scoring: dict):
    forbidden = [str(f).lower() for f in scoring.get("forbidden", [])]
    found = [f for f in forbidden if f in response.lower()]
    if not found:
        return 1.0, "No forbidden words found"
    return 0.0, f"Forbidden words found: {found}"


def _score_ends_with(response: str, scoring: dict):
    expected = str(scoring.get("answer", scoring.get("value", ""))).strip().lower()
    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    if not lines:
        return 0.0, "Empty response"
    last_line = lines[-1].rstrip(".,!?;:").lower()
    last_word = last_line.split()[-1] if last_line.split() else ""
    if last_word == expected:
        return 1.0, f"Response ends with '{expected}'"
    return 0.0, f"Last word: '{last_word}', expected '{expected}'"


def _score_word_count(response: str, scoring: dict):
    words = response.strip().split()
    count = len(words)
    min_w = int(scoring.get("min", 0))
    max_w = int(scoring.get("max", 10 ** 9))
    if min_w <= count <= max_w:
        return 1.0, f"{count} words (required {min_w}–{max_w})"
    return 0.0, f"{count} words, required {min_w}–{max_w}"


def _score_regex(response: str, scoring: dict):
    pattern = scoring["pattern"]
    if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
        return 1.0, "Regex matched"
    return 0.0, f"Regex not matched: {pattern}"


def _score_json_keys(response: str, scoring: dict):
    required = scoring.get("keys", [])
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return 0.0, "No JSON object found in response"
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as e:
        return 0.0, f"Invalid JSON: {e}"
    missing = [k for k in required if k not in data]
    if not missing:
        return 1.0, f"All required keys present: {required}"
    return 0.0, f"Missing keys: {missing}"


def _score_line_count(response: str, scoring: dict):
    expected = int(scoring["count"])
    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    if len(lines) == expected:
        return 1.0, f"{len(lines)} lines — correct"
    return 0.0, f"Got {len(lines)} lines, expected {expected}"


def _score_code_exec(response: str, scoring: dict):
    code = _extract_code(response)
    test_code = scoring.get("test_code", "print('PASS')")
    full_code = code + "\n\n" + test_code
    tmp: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_code)
            tmp = f.name

        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if proc.returncode == 0 and "PASS" in proc.stdout:
            return 1.0, "All assertions passed"
        if proc.returncode == 0:
            return 0.5, f"Ran without error but no PASS marker: {proc.stdout[:120]}"
        return 0.0, f"Runtime error: {proc.stderr[:300]}"

    except subprocess.TimeoutExpired:
        return 0.0, "Execution timed out (10 s)"
    except Exception as e:
        return 0.0, f"Exec error: {e}"
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
