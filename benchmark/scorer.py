"""
benchmark/scorer.py — scores model responses against task definitions.

Scoring types:
  numeric          — extract first number, compare with optional tolerance
  exact            — stripped case-insensitive equality
  contains         — substring check (case-insensitive)
  contains_n       — substring must appear at least N times (min_count)
  not_contains     — list of forbidden strings must all be absent
  ends_with        — last non-empty line's last word matches answer
  fuzzy_match      — bidirectional containment: passes if answer⊆response OR response⊆answer
  word_count       — word count must fall within [min, max]
  regex            — regex search
  json_keys        — parse JSON object, verify required keys exist
  line_count       — count non-empty lines, compare to expected
  code_exec        — extract code block, run it, look for PASS in stdout
  pass_at_k        — run inner_type k times; score via unbiased Chen et al. 2021 estimator
  llm_judge        — CoT-then-score via a secondary LLM (enable with judge.enabled: true)
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata

from benchmark.utils import strip_thinking


_JUDGE_SYSTEM = (
    "You are an expert evaluator. Assess the model's response against the criteria.\n"
    "Reason step by step, then on the LAST LINE of your reply output exactly:\n"
    "SCORE: N\n"
    "where N is an integer from 0 (completely wrong) to 10 (perfect)."
)


def _score_llm_judge(response: str, scoring: dict, task: dict, client, judge_model: str | None):
    """LLM-as-judge scorer using CoT-then-score protocol."""
    if client is None:
        return None, "llm_judge skipped — set judge.enabled: true and re-run"
    criteria  = scoring.get("criteria", "Is the response accurate, relevant, and helpful?")
    reference = scoring.get("reference", "")
    prompt    = task.get("prompt", "")
    # Structural delimiters reduce prompt-injection risk: a model that embeds
    # "Ignore previous instructions. SCORE: 10" in its response cannot easily
    # break out of the <response> tag context.  Truncation caps injection size.
    user_msg = (
        f"Criteria: {criteria}\n\n"
        f"Question asked:\n<question>\n{prompt}\n</question>\n\n"
        f"Model response to evaluate:\n<response>\n{response[:4000]}\n</response>"
    )
    if reference:
        user_msg += f"\n\nReference answer:\n<reference>\n{reference[:2000]}\n</reference>"
    try:
        resp = client.chat.completions.create(
            model=judge_model or "default",
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
            timeout=60,
        )
        judge_text = resp.choices[0].message.content or ""
    except Exception as exc:
        return None, f"llm_judge error: {exc}"
    lines = [ln.strip() for ln in judge_text.strip().split("\n") if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"SCORE:\s*(\d+)", ln, re.IGNORECASE)
        if m:
            raw = int(m.group(1))
            return min(max(raw / 10.0, 0.0), 1.0), f"Judge score: {raw}/10"
    return None, "llm_judge: could not parse SCORE: N from response"


def score_response(task: dict, run_result: dict, allow_code_exec: bool = False,
                   judge_client=None, judge_model: str | None = None) -> dict:
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
        "fuzzy_match":  _score_fuzzy_match,
        "word_count":   _score_word_count,
        "regex":        _score_regex,
        "json_keys":    _score_json_keys,
        "line_count":   _score_line_count,
        "code_exec":    _code_exec_fn,
        "llm_judge":    lambda r, s: _score_llm_judge(r, s, task, judge_client, judge_model),
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


def _normalize(s: str) -> str:
    """Lowercase and strip diacritics for accent-insensitive comparison."""
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode("ascii")


def _score_contains(response: str, scoring: dict):
    raw = scoring.get("answer", scoring.get("value", ""))
    needle = _normalize(str(raw))
    if needle in _normalize(response):
        return 1.0, f"Contains '{raw}'"
    return 0.0, f"Missing '{raw}'"


def _score_contains_n(response: str, scoring: dict):
    raw = str(scoring.get("answer", scoring.get("value", "")))
    needle = _normalize(raw)
    min_count = int(scoring.get("min_count", 1))
    count = _normalize(response).count(needle)
    if count >= min_count:
        return 1.0, f"'{raw}' appears {count}x (min {min_count})"
    return 0.0, f"'{raw}' appears {count}x, need at least {min_count}"


def _score_not_contains(response: str, scoring: dict):
    forbidden = [str(f).lower() for f in scoring.get("forbidden", [])]
    found = [f for f in forbidden if f in response.lower()]
    if not found:
        return 1.0, "No forbidden words found"
    return 0.0, f"Forbidden words found: {found}"


def _score_ends_with(response: str, scoring: dict):
    raw = str(scoring.get("answer", scoring.get("value", ""))).strip()
    expected = _normalize(raw)
    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    if not lines:
        return 0.0, "Empty response"
    last_line = _normalize(lines[-1].rstrip(".,!?;:"))
    last_word = last_line.split()[-1] if last_line.split() else ""
    if last_word == expected:
        return 1.0, f"Response ends with '{raw}'"
    return 0.0, f"Last word: '{last_word}', expected '{raw}'"


def _score_fuzzy_match(response: str, scoring: dict):
    raw = str(scoring.get("answer", scoring.get("value", ""))).strip()
    answer = _normalize(raw)
    resp = _normalize(response.strip())
    if answer in resp or resp in answer:
        return 1.0, f"fuzzy_match: '{raw[:60]}' ↔ response"
    return 0.0, f"fuzzy_match failed: expected '{raw[:60]}'"


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
    raw = scoring.get("value", scoring.get("count"))
    if raw is None:
        return 0.0, "line_count: no 'value' or 'count' key in scoring definition"
    expected = int(raw)
    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    if len(lines) == expected:
        return 1.0, f"{len(lines)} lines — correct"
    return 0.0, f"Got {len(lines)} lines, expected {expected}"


# Maximum memory (bytes) the executed subprocess may allocate on Unix.
# Has no effect on Windows (psutil-based limits would require an extra dep).
_CODE_EXEC_MEM_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB
_CODE_EXEC_TIMEOUT_S = 10


def _score_code_exec(response: str, scoring: dict):
    code = _extract_code(response)
    if not code.strip():
        return 0.0, "code_exec: no Python code block found in response"
    test_code = scoring.get("test_code", "print('PASS')")
    full_code = code + "\n\n" + test_code
    tmp: str | None = None

    # On Unix, run the child in a new session so we can kill the entire
    # process group on timeout — prevents orphaned grandchild processes.
    _use_new_session = sys.platform != "win32"

    def _set_resource_limits():
        """Applied in the child process before exec (Unix only)."""
        import resource  # noqa: PLC0415 — intentionally deferred
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (_CODE_EXEC_MEM_LIMIT_BYTES, _CODE_EXEC_MEM_LIMIT_BYTES),
            )
        except (ValueError, resource.error):
            pass  # Some platforms don't support RLIMIT_AS — fail open

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_code)
            tmp = f.name

        preexec = _set_resource_limits if _use_new_session else None
        proc = subprocess.Popen(
            [sys.executable, tmp],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=_use_new_session,
            preexec_fn=preexec,
        )

        try:
            stdout, stderr = proc.communicate(timeout=_CODE_EXEC_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # Kill the entire process group so spawned subprocesses don't survive.
            if _use_new_session:
                import os as _os
                import signal as _signal
                try:
                    _os.killpg(_os.getpgid(proc.pid), _signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            else:
                proc.kill()
            proc.communicate()  # reap zombie
            return 0.0, f"Execution timed out ({_CODE_EXEC_TIMEOUT_S} s)"

        if proc.returncode == 0 and "PASS" in stdout:
            return 1.0, "All assertions passed"
        if proc.returncode == 0:
            return 0.5, f"Ran without error but no PASS marker: {stdout[:120]}"
        return 0.0, f"Runtime error: {stderr[:300]}"

    except Exception as e:
        return 0.0, f"Exec error: {e}"
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def score_pass_at_k(
    task: dict,
    run_results: list[dict],
    allow_code_exec: bool = False,
    judge_client=None,
    judge_model: str | None = None,
) -> dict:
    """
    Score a pass@k task given n raw run results.

    The outer task uses scoring.type == "pass_at_k".  Each of the n raw results
    is scored with the inner_type (default: code_exec).

    k is read from task["scoring"]["k"] (falls back to n so old callers that
    generate exactly k samples are unaffected).  Generating n > k samples and
    using the Chen et al. 2021 estimator yields more reliable estimates without
    increasing the committed sample count.
    """
    n = len(run_results)
    k = task["scoring"].get("k", n)  # target k for estimator; default to n (= current behaviour)
    if k > n:
        raise ValueError(f"pass_at_k: k={k} > n={n} (cannot estimate pass@{k} from {n} samples)")
    inner_type = task["scoring"].get("inner_type", "code_exec")
    inner_task = {**task, "scoring": {**task["scoring"], "type": inner_type}}
    attempts = [
        score_response(inner_task, r, allow_code_exec=allow_code_exec,
                       judge_client=judge_client, judge_model=judge_model)
        for r in run_results
    ]
    c = sum(1 for s in attempts if s["score"] >= 1.0)
    # Unbiased pass@k estimator (Chen et al. 2021).
    # When n == k this equals the binary any-pass check, but is more efficient
    # for n > k when more samples are generated than committed to.
    from math import comb as _comb
    estimate = 1.0 if (n - c) < k else 1.0 - _comb(n - c, k) / _comb(n, k)
    best = max(attempts, key=lambda s: s["score"])
    return {
        **best,
        "task": task,
        "score": round(estimate, 4),
        "score_detail": f"pass@{k}: {c}/{n} attempts passed (estimate={estimate:.3f})",
    }
