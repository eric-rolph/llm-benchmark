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
  repo_patch       — apply model edits to a fixture repo, inject hidden tests, run tests
  agent_loop       — observed repo tool-use loop scored by hidden tests
  logprob_choice   — compare highest-probability token against expected answer (for base models)
  pass_at_k        — run inner_type k times; score via unbiased Chen et al. 2021 estimator
  workflow_trace   — verify JSON tool-call trace plus final/replayed mock state
  llm_judge        — CoT-then-score via a secondary LLM (enable with judge.enabled: true)
  rubric_judge     — decomposed multi-criterion rubric via LLM (XpertBench ShotJudge pattern)
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from copy import deepcopy

from benchmark.evaluation import annotate_pass, result_passed
from benchmark.repo_patch import score_repo_patch
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


# ── rubric judge ─────────────────────────────────────────────────────────────

_RUBRIC_SYSTEM = (
    "You are a rigorous grader. Evaluate the response criterion-by-criterion.\n"
    "For each criterion, output a line in EXACTLY this format:\n"
    "  CRITERION <n>: <PASS|PARTIAL|FAIL> (<brief reason>)\n"
    "Replace <n> with the criterion number starting at 1.\n"
    "PASS = fully satisfied, PARTIAL = partially satisfied, FAIL = not satisfied.\n"
    "After all criteria, output one final line:\n"
    "  RUBRIC_SCORE: <0.0–1.0>\n"
    "where the score is the weighted average you computed from the criteria."
)

_RUBRIC_SCALE = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}


def _score_rubric_judge(response: str, scoring: dict, task: dict, client, judge_model: str | None):
    """
    Decomposed rubric scorer (XpertBench ShotJudge pattern).

    scoring schema:
      type: rubric_judge
      criteria:
        - criterion: "The response directly answers the question"
          weight: 2.0
        - criterion: "Key facts are accurate"
          weight: 3.0
        - criterion: "Response is appropriately concise"
          weight: 1.0
      reference: "optional reference answer"   # optional
    """
    if client is None:
        return None, "rubric_judge skipped — set judge.enabled: true and re-run"

    criteria_list = scoring.get("criteria", [])
    if not criteria_list:
        return None, "rubric_judge: no 'criteria' list in scoring definition"

    reference = scoring.get("reference", "")
    prompt    = task.get("prompt", "")

    # Build numbered criteria block
    criteria_block = "\n".join(
        f"  {i+1}. [{float(c.get('weight', 1.0)):.1f}x] {c['criterion']}"
        for i, c in enumerate(criteria_list)
    )

    user_msg = (
        f"Evaluate the following response using the rubric below.\n\n"
        f"Task prompt:\n<question>\n{prompt}\n</question>\n\n"
        f"Rubric criteria (weight shown as multiplier):\n{criteria_block}\n\n"
        f"Response to evaluate:\n<response>\n{response[:4000]}\n</response>"
    )
    if reference:
        user_msg += f"\n\nReference answer (for factual grounding):\n<reference>\n{reference[:2000]}\n</reference>"

    try:
        resp = client.chat.completions.create(
            model=judge_model or "default",
            messages=[
                {"role": "system", "content": _RUBRIC_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=768,
            timeout=90,
        )
        judge_text = resp.choices[0].message.content or ""
    except Exception as exc:
        return None, f"rubric_judge error: {exc}"

    # Parse RUBRIC_SCORE from last line (most reliable signal)
    lines = [ln.strip() for ln in judge_text.strip().split("\n") if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"RUBRIC_SCORE:\s*([01](?:\.\d+)?)", ln, re.IGNORECASE)
        if m:
            score = float(m.group(1))
            score = min(max(score, 0.0), 1.0)

            # Parse per-criterion verdicts for the detail string
            verdicts = re.findall(r"CRITERION\s+\d+:\s*(PASS|PARTIAL|FAIL)", judge_text, re.IGNORECASE)
            detail = f"Rubric: {' | '.join(v.upper() for v in verdicts)}  →  {score:.2f}"
            return score, detail

    # Fallback: compute weighted score from criterion lines if RUBRIC_SCORE missing
    verdicts = re.findall(r"CRITERION\s+(\d+):\s*(PASS|PARTIAL|FAIL)", judge_text, re.IGNORECASE)
    if verdicts and len(verdicts) == len(criteria_list):
        w_sum = 0.0
        w_tot = 0.0
        for idx_str, verdict in verdicts:
            idx = int(idx_str) - 1
            if 0 <= idx < len(criteria_list):
                w = float(criteria_list[idx].get("weight", 1.0))
                w_sum += _RUBRIC_SCALE.get(verdict.upper(), 0.0) * w
                w_tot += w
        if w_tot > 0:
            score = w_sum / w_tot
            detail = f"Rubric (computed): {' | '.join(v.upper() for _, v in verdicts)}  →  {score:.2f}"
            return min(max(score, 0.0), 1.0), detail

    return None, "rubric_judge: could not parse RUBRIC_SCORE or criterion verdicts"


def score_response(task: dict, run_result: dict, allow_code_exec: bool = False,
                   judge_client=None, judge_model: str | None = None) -> dict:
    scored = {
        **run_result,
        "task": task,
        "score": 0.0,
        "max_score": 1.0,
        "score_detail": "",
    }
    annotate_pass(scored)

    if run_result.get("error"):
        scored["score_detail"] = f"API error: {run_result['error']}"
        return scored

    # Strip any think/reasoning blocks that leaked into content
    response = strip_thinking(run_result["response"])
    scoring = task.get("scoring", {})
    method = scoring.get("type", "contains")

    if method == "agent_loop":
        scored["score"] = float(run_result.get("agent_loop_score", 0.0) or 0.0)
        scored["score_detail"] = run_result.get("agent_loop_detail", "agent_loop: no detail")
        return annotate_pass(scored)

    _code_exec_fn = _score_code_exec if allow_code_exec else (
        lambda r, s: (0.0, "code_exec disabled — rerun with --allow-code-exec")
    )
    _repo_patch_fn = score_repo_patch if allow_code_exec else (
        lambda r, s: (0.0, "repo_patch disabled — rerun with --allow-code-exec")
    )

    dispatch = {
        "numeric":          _score_numeric,
        "exact":            _score_exact,
        "contains":         _score_contains,
        "contains_n":       _score_contains_n,
        "multi_contains":   _score_multi_contains,
        "not_contains":     _score_not_contains,
        "ends_with":        _score_ends_with,
        "fuzzy_match":      _score_fuzzy_match,
        "word_count":       _score_word_count,
        "regex":            _score_regex,
        "json_keys":        _score_json_keys,
        "json_schema":      _score_json_schema,
        "line_count":       _score_line_count,
        "code_exec":        _code_exec_fn,
        "repo_patch":       _repo_patch_fn,
        "logprob_choice":   _score_logprob_choice,
        "workflow_trace":   _score_workflow_trace,
        "llm_judge":        lambda r, s: _score_llm_judge(r, s, task, judge_client, judge_model),
        "rubric_judge":     lambda r, s: _score_rubric_judge(r, s, task, judge_client, judge_model),
    }

    fn = dispatch.get(method, lambda r, s: (0.0, f"Unknown scoring type: {method}"))
    score, detail = fn(response, scoring)
    scored["score"] = float(score) if score is not None else 0.0
    scored["score_detail"] = detail
    return annotate_pass(scored)


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_numbers(text: str) -> list[float]:
    """All numeric values in arbitrary text, commas stripped."""
    numbers = []
    for m in re.findall(r"-?\d+\.?\d*", text.replace(",", "")):
        try:
            numbers.append(float(m))
        except ValueError:
            pass
    return numbers


def _extract_code(response: str, extract: str = "first") -> str:
    """Extract a Python code block from a model response.

    extract="first" returns the first fenced block (default).
    extract="last"  returns the last fenced block — useful for
    self-critique tasks where the IMPROVED solution comes last.
    """
    python_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    if python_blocks:
        return (python_blocks[-1] if extract == "last" else python_blocks[0]).strip()
    all_blocks = re.findall(r"```\n?(.*?)```", response, re.DOTALL)
    if all_blocks:
        return (all_blocks[-1] if extract == "last" else all_blocks[0]).strip()
    # Heuristic: start from the first def / class / import line
    lines = response.strip().split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^(def |class |import |from )", line):
            return "\n".join(lines[i:]).strip()
    return response.strip()


def _extract_json_value(response: str):
    """Extract the first JSON value from a response, including fenced JSON."""
    text = response.strip()
    candidates = [text]

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in ("{", "["):
            continue
        try:
            data, _ = decoder.raw_decode(text[idx:])
            return data
        except json.JSONDecodeError:
            continue
    return None


_MISSING = object()


def _get_path(data, dotted_path: str):
    """Read a dotted path from dict/list data; numeric segments index lists."""
    current = data
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return _MISSING
            continue
        return _MISSING
    return current


def _call_name(call) -> str:
    if isinstance(call, str):
        return call
    if isinstance(call, dict):
        for key in ("tool", "name", "action", "type"):
            if call.get(key):
                return str(call[key])
    return ""


def _call_args(call) -> dict:
    """Return normalized tool-call arguments from common agent trace shapes."""
    if not isinstance(call, dict):
        return {}

    for key in ("args", "arguments", "parameters"):
        value = call.get(key)
        if isinstance(value, dict):
            return value

    return {
        key: value
        for key, value in call.items()
        if key not in {"tool", "name", "action", "type"}
    }


def _is_subsequence(needles: list[str], haystack: list[str]) -> bool:
    pos = 0
    for name in haystack:
        if pos < len(needles) and name == needles[pos]:
            pos += 1
    return pos == len(needles)


def _contains_value(actual, expected) -> bool:
    if actual is _MISSING:
        return False
    if isinstance(actual, str):
        return _normalize(str(expected)) in _normalize(actual)
    if isinstance(actual, list):
        if isinstance(expected, list):
            return all(
                any(_matches_expected(actual_item, expected_item) for actual_item in actual)
                for expected_item in expected
            )
        return any(_matches_expected(actual_item, expected) for actual_item in actual)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return all(actual.get(k) == v for k, v in expected.items())
    return actual == expected


def _matches_expected(actual, expected) -> bool:
    if isinstance(expected, dict) and set(expected) == {"contains"}:
        return _contains_value(actual, expected["contains"])
    if isinstance(expected, dict) and set(expected) == {"regex"}:
        return actual is not _MISSING and re.search(str(expected["regex"]), str(actual), re.IGNORECASE) is not None
    return actual == expected


_ARG_TOKEN_RE = re.compile(r"\$args\.([A-Za-z_][A-Za-z0-9_]*)")


def _resolve_arg_templates(value, args: dict):
    """Resolve $args.name templates inside replay effect values."""
    if isinstance(value, str):
        matches = _ARG_TOKEN_RE.findall(value)
        if not matches:
            return value, []

        missing = [name for name in matches if name not in args]
        if missing:
            return value, missing

        full = _ARG_TOKEN_RE.fullmatch(value)
        if full:
            return args[full.group(1)], []

        resolved = _ARG_TOKEN_RE.sub(lambda m: str(args[m.group(1)]), value)
        return resolved, []

    if isinstance(value, list):
        resolved = []
        missing = []
        for item in value:
            item_value, item_missing = _resolve_arg_templates(item, args)
            resolved.append(item_value)
            missing.extend(item_missing)
        return resolved, missing

    if isinstance(value, dict):
        resolved = {}
        missing = []
        for key, item in value.items():
            key_value, key_missing = _resolve_arg_templates(str(key), args)
            item_value, item_missing = _resolve_arg_templates(item, args)
            resolved[key_value] = item_value
            missing.extend(key_missing)
            missing.extend(item_missing)
        return resolved, missing

    return value, []


def _set_path(data: dict, dotted_path: str, value) -> str | None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        return "empty state path"
    current = data
    for part in parts[:-1]:
        if not isinstance(current, dict):
            return f"cannot set through non-object path segment {part!r}"
        current = current.setdefault(part, {})
    if not isinstance(current, dict):
        return f"cannot set state path {dotted_path!r}"
    current[parts[-1]] = value
    return None


def _ensure_list_path(data: dict, dotted_path: str):
    current = _get_path(data, dotted_path)
    if current is not _MISSING:
        return current if isinstance(current, list) else _MISSING
    error = _set_path(data, dotted_path, [])
    if error:
        return _MISSING
    return _get_path(data, dotted_path)


def _replay_workflow_state(calls: list, replay: dict) -> tuple[dict, list[str]]:
    """Derive final mock state by replaying configured tool effects."""
    state = deepcopy(replay.get("initial_state", {}))
    effects = replay.get("effects", {})
    failures: list[str] = []

    if not isinstance(state, dict):
        return {}, ["workflow replay initial_state must be an object"]
    if not isinstance(effects, dict) or not effects:
        return state, ["workflow replay has no effects configured"]

    for index, call in enumerate(calls):
        tool = _call_name(call)
        effect = effects.get(tool)
        if not effect:
            continue

        args = _call_args(call)
        missing_required = [name for name in effect.get("required_args", []) if name not in args]
        if missing_required:
            failures.append(f"call {index} {tool!r} missing arg(s): {missing_required}")
            continue

        for raw_path, raw_value in effect.get("set", {}).items():
            path, missing_path = _resolve_arg_templates(str(raw_path), args)
            value, missing_value = _resolve_arg_templates(raw_value, args)
            missing = sorted(set(missing_path + missing_value))
            if missing:
                failures.append(f"call {index} {tool!r} could not resolve arg(s): {missing}")
                continue
            error = _set_path(state, str(path), value)
            if error:
                failures.append(f"call {index} {tool!r}: {error}")

        for raw_path, raw_value in effect.get("append", {}).items():
            path, missing_path = _resolve_arg_templates(str(raw_path), args)
            value, missing_value = _resolve_arg_templates(raw_value, args)
            missing = sorted(set(missing_path + missing_value))
            if missing:
                failures.append(f"call {index} {tool!r} could not resolve arg(s): {missing}")
                continue
            target = _ensure_list_path(state, str(path))
            if target is _MISSING:
                failures.append(f"call {index} {tool!r}: state.{path} is not a list")
                continue
            target.append(value)

    return state, failures


# ── scorers ──────────────────────────────────────────────────────────────────

def _score_workflow_trace(response: str, scoring: dict):
    """
    Deterministically verify a model-produced workflow trace.

    Expected response shape:
      {
        "tool_calls": [{"tool": "read_file"}, {"tool": "edit_file"}],
        "state": {"repo": {"tests_passed": true}}
      }

    Scoring YAML keys:
      required_tools  — every tool name must appear at least once
      ordered_tools   — tool names must appear as an ordered subsequence
      forbidden_tools — tool names that must not appear
      required_call_args — tool calls that must include specific args
      min_calls/max_calls — bounds on the tool call count
      replay — derive state by applying configured tool effects
      expected_state  — dotted paths under state/final_state with exact values
      state_contains  — dotted paths under state/final_state containing a value
    """
    data = _extract_json_value(response)
    if not isinstance(data, dict):
        return 0.0, "workflow_trace: response did not contain a JSON object"

    calls_field = scoring.get("calls_field", "tool_calls")
    state_field = scoring.get("state_field", "state")
    calls = data.get(calls_field, data.get("calls", []))
    if not isinstance(calls, list):
        return 0.0, f"workflow_trace: {calls_field!r} must be a list"

    state = data.get(state_field, data.get("final_state", {}))
    if state is None:
        state = {}
    if not isinstance(state, dict):
        return 0.0, f"workflow_trace: {state_field!r} must be an object"

    call_names = [_call_name(call) for call in calls]
    replay = scoring.get("replay")
    replay_failures: list[str] = []
    if replay:
        if not isinstance(replay, dict):
            return 0.0, "workflow_trace: replay must be an object"
        state, replay_failures = _replay_workflow_state(calls, replay)

    failures: list[str] = []
    total = 0
    passed = 0

    def check(condition: bool, failure: str) -> None:
        nonlocal total, passed
        total += 1
        if condition:
            passed += 1
        else:
            failures.append(failure)

    if "min_calls" in scoring:
        expected = int(scoring["min_calls"])
        check(len(calls) >= expected, f"expected at least {expected} call(s), got {len(calls)}")
    if "max_calls" in scoring:
        expected = int(scoring["max_calls"])
        check(len(calls) <= expected, f"expected at most {expected} call(s), got {len(calls)}")

    for replay_failure in replay_failures:
        check(False, replay_failure)

    for tool in scoring.get("required_tools", []):
        check(str(tool) in call_names, f"missing required tool {tool!r}")

    for tool in scoring.get("forbidden_tools", []):
        check(str(tool) not in call_names, f"forbidden tool used: {tool!r}")

    for requirement in scoring.get("required_call_args", []):
        tool = str(requirement.get("tool", ""))
        expected_args = requirement.get("args", {})
        matching_call = False
        for call in calls:
            if _call_name(call) != tool:
                continue
            args = _call_args(call)
            if all(_matches_expected(_get_path(args, str(path)), expected) for path, expected in expected_args.items()):
                matching_call = True
                break
        check(matching_call, f"missing call {tool!r} with args {expected_args!r}")

    ordered = [str(tool) for tool in scoring.get("ordered_tools", [])]
    if ordered:
        check(_is_subsequence(ordered, call_names), f"tool order missing subsequence {ordered!r}")

    for path, expected in scoring.get("expected_state", {}).items():
        actual = _get_path(state, str(path))
        check(
            _matches_expected(actual, expected),
            f"state.{path} expected {expected!r}, got {None if actual is _MISSING else actual!r}",
        )

    for path, expected in scoring.get("state_contains", {}).items():
        actual = _get_path(state, str(path))
        check(
            _contains_value(actual, expected),
            f"state.{path} did not contain {expected!r}",
        )

    if total == 0:
        return 0.0, "workflow_trace: no checks configured"

    score = passed / total
    if not failures:
        return 1.0, f"workflow_trace: all {total} checks passed"
    detail = f"workflow_trace: {passed}/{total} checks passed; " + "; ".join(failures[:4])
    return score, detail


def _score_logprob_choice(response: str, scoring: dict):
    """
    Score a logprob-based multiple choice response.

    The runner already selected the highest-probability token and placed it
    in the 'response' field.  We just do a case-insensitive exact match
    against the expected answer (e.g. "A", "B", "C", "D").
    """
    raw = scoring.get("answer", scoring.get("value", ""))
    expected = str(raw).strip().upper()
    got = response.strip().upper()
    if got == expected:
        return 1.0, f"Logprob match: {got}"
    return 0.0, f"Logprob: got '{got}', expected '{expected}'"


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
    # Pick the number closest to expected — avoids false positives when the
    # prompt itself contains a number (e.g. "Carbon-14" appearing before the
    # real answer "5730").
    candidates = _extract_numbers(response)
    if not candidates:
        return 0.0, f"No number found. Expected {expected}"
    got = min(candidates, key=lambda n: abs(n - expected))
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
    if not needle:
        # "" is a substring of everything — a misconfigured task must not score 1.0
        return 0.0, f"contains: empty answer/value in scoring definition ({raw!r})"
    if needle in _normalize(response):
        return 1.0, f"Contains '{raw}'"
    return 0.0, f"Missing '{raw}'"


def _score_contains_n(response: str, scoring: dict):
    raw = str(scoring.get("answer", scoring.get("value", "")))
    needle = _normalize(raw)
    if not needle:
        return 0.0, f"contains_n: empty answer/value in scoring definition ({raw!r})"
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
    if not answer:
        return 0.0, f"fuzzy_match: empty answer/value in scoring definition ({raw!r})"
    resp = _normalize(response.strip())
    if not resp:
        # "" ⊂ answer would pass below — an empty response is a fail, not a match
        return 0.0, "fuzzy_match: empty response"
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
    pattern = scoring.get("pattern")
    if not pattern:
        return 0.0, "regex: no 'pattern' key in scoring definition"
    try:
        matched = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    except re.error as e:
        return 0.0, f"regex: invalid pattern {pattern!r}: {e}"
    if matched:
        return 1.0, "Regex matched"
    return 0.0, f"Regex not matched: {pattern}"


def _score_json_keys(response: str, scoring: dict):
    required = scoring.get("keys", [])
    decoder = json.JSONDecoder()
    best_missing = None

    for idx, char in enumerate(response):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(response[idx:])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue

        missing = [k for k in required if k not in data]
        if not missing:
            return 1.0, f"All required keys present: {required}"
        if best_missing is None or len(missing) < len(best_missing):
            best_missing = missing

    if best_missing is None:
        return 0.0, "No JSON object found in response"
    return 0.0, f"Missing keys: {best_missing}"


def _score_json_schema(response: str, scoring: dict):
    """Validate response JSON against a lightweight inline schema.

    Scoring YAML keys:
      root         — "array" or "object" (default: "object")
      min_items    — minimum array length when root=array (default: 0)
      required_keys — keys that must be present on each item (array) or root (object)
      array_keys   — keys whose values must themselves be arrays (object root only)
      expected_values — dotted JSON paths with exact/contains/regex expected values
      expected_items — partial objects that must be present when root=array
    """
    root_type     = scoring.get("root", "object")
    required_keys = scoring.get("required_keys", [])
    array_keys    = scoring.get("array_keys", [])
    min_items     = int(scoring.get("min_items", 0))

    # Strip markdown fences so models that wrap JSON in ```json ... ``` still pass.
    text = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(response):
            if ch in ("{", "["):
                try:
                    data, _ = decoder.raw_decode(response[i:])
                    break
                except json.JSONDecodeError:
                    continue

    if data is None:
        return 0.0, "No valid JSON found in response"

    if root_type == "array":
        if not isinstance(data, list):
            return 0.0, f"Expected JSON array at root, got {type(data).__name__}"
        if len(data) < min_items:
            return 0.0, f"Array has {len(data)} item(s), need ≥{min_items}"
        items = data
    else:
        if not isinstance(data, dict):
            return 0.0, f"Expected JSON object at root, got {type(data).__name__}"
        items = [data]
        errors = [f"Key {k!r} must be an array" for k in array_keys
                  if k in data and not isinstance(data[k], list)]
        if errors:
            return 0.0, "; ".join(errors)

    if required_keys:
        item_errors = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                item_errors.append(f"Item {idx} is not an object")
                continue
            missing = [k for k in required_keys if k not in item]
            if missing:
                item_errors.append(f"Item {idx} missing: {missing}")
        if item_errors:
            return 0.0, "; ".join(item_errors[:3])

    value_errors = []
    for path, expected in scoring.get("expected_values", {}).items():
        actual = _get_path(data, str(path))
        if not _matches_expected(actual, expected):
            got = None if actual is _MISSING else actual
            value_errors.append(f"{path} expected {expected!r}, got {got!r}")
    if value_errors:
        return 0.0, "; ".join(value_errors[:3])

    expected_items = scoring.get("expected_items", [])
    if expected_items:
        if not isinstance(data, list):
            return 0.0, "expected_items requires JSON array root"
        item_errors = []
        for expected in expected_items:
            if not isinstance(expected, dict):
                item_errors.append(f"expected item is not an object: {expected!r}")
                continue
            found = any(
                isinstance(item, dict)
                and all(
                    _matches_expected(item.get(key, _MISSING), value)
                    for key, value in expected.items()
                )
                for item in data
            )
            if not found:
                item_errors.append(f"missing expected item: {expected!r}")
        if item_errors:
            return 0.0, "; ".join(item_errors[:3])

    n = len(items)
    return 1.0, f"JSON schema valid ({n} item{'s' if n != 1 else ''})"


def _score_multi_contains(response: str, scoring: dict):
    """
    Require all groups to match (case-insensitive substring).

    Each group is a list of alternatives — the response must contain at least
    one item from EVERY group (AND across groups, OR within each group).
    'values' is a convenience shorthand: a flat list where ALL must appear.

    YAML examples:
      type: multi_contains
      values: ["battery", "stanford"]

      type: multi_contains
      groups:
        - ["battery", "charg"]
        - ["stanford", "fast", "new", "rapid"]
    """
    resp_norm = _normalize(response)
    groups = scoring.get("groups")
    if groups is None:
        values = scoring.get("values", [])
        groups = [[v] for v in values]
    if not groups:
        return 0.0, "multi_contains: no 'groups' or 'values' defined"
    missing = []
    for group in groups:
        needles = [_normalize(str(v)) for v in group]
        if not any(n in resp_norm for n in needles):
            missing.append(group)
    if not missing:
        return 1.0, f"All {len(groups)} group(s) matched"
    return 0.0, f"Missing matches for: {missing}"


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
    code = _extract_code(response, extract=scoring.get("extract", "first"))
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
        # Untrusted code must not inherit secrets (API keys etc.) from the
        # benchmark process — pass a minimal whitelisted environment and run
        # outside the repo so relative paths can't touch it.
        _env_keep = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
                     "PATHEXT", "TEMP", "TMP", "LANG", "LC_ALL", "PYTHONIOENCODING")
        child_env = {k: v for k, v in os.environ.items() if k.upper() in _env_keep}
        proc = subprocess.Popen(
            [sys.executable, tmp],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=_use_new_session,
            preexec_fn=preexec,
            env=child_env,
            cwd=tempfile.gettempdir(),
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
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                try:
                    proc.kill()
                except OSError:
                    pass
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
    c = sum(1 for s in attempts if result_passed(s))
    # Unbiased pass@k estimator (Chen et al. 2021).
    # When n == k this equals the binary any-pass check, but is more efficient
    # for n > k when more samples are generated than committed to.
    from math import comb as _comb
    estimate = 1.0 if (n - c) < k else 1.0 - _comb(n - c, k) / _comb(n, k)
    best = max(attempts, key=lambda s: s["score"])
    result = {
        **best,
        "task": task,
        "score": round(estimate, 4),
        "score_detail": f"pass@{k}: {c}/{n} attempts passed (estimate={estimate:.3f})",
    }
    for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "api_cost"):
        total = _sum_numeric_result_field(run_results, field)
        if total is not None:
            result[field] = total
    return annotate_pass(result)


def _sum_numeric_result_field(results: list[dict], field: str) -> int | float | None:
    values = []
    for result in results:
        raw = result.get(field)
        if raw in (None, ""):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    total = sum(values)
    if field != "api_cost" and total.is_integer():
        return int(total)
    if field == "api_cost":
        return round(total, 8)
    return total
