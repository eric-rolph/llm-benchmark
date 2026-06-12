# LLM Benchmark — Backlog

Prioritised list of improvements derived from benchmark runs, the
akitaonrails coding-benchmark article, the qwopus36-eval review, and
the OpenAI SWE-bench Verified post-mortem.

---

## Tier 1 — Bug fixes (wrong results today)

### ✅ 1.1 `headline_from_article` regex is too narrow
**Fixed:** Replaced narrow ordered regex with `multi_contains` scorer using two
groups — `["battery","charg"]` AND `["stanford","smartphone","fast","rapid","new","speed","minute"]`.
Required adding `_score_multi_contains` to `benchmark/scorer.py`.  
**Files:** `tasks/summarization.yaml`, `benchmark/scorer.py`

### ✅ 1.2 `vision_describe_dog` keyword is too fragile
**Fixed:** Changed scorer from `contains: dog` to `regex: (?i)(dog|canine|puppy|pup)`.  
**File:** `tasks/vision.yaml`

### ✅ 1.3 Qwen3 thinking-mode code block extraction fails
**Fixed:** When `response_text` is empty after `strip_thinking` but `reasoning_text`
is non-empty, fall back to `strip_thinking(reasoning_text)`. Handles models like
Qwen3 that route all output through `reasoning_content`.  
**File:** `benchmark/runner.py`

---

## Tier 2 — Task quality

### ✅ 2.1 Add `contamination_risk` field to LeetCode-style tasks
**Fixed:** Added `contamination_risk: high` to code_001–005, code_007, code_011
(LIS, group anagrams, max subarray, count islands, two-sum, merge intervals,
rotate matrix). Reporter now prints an "Excl. memorised" row in the accuracy
table that filters these out.  
**Files:** `tasks/coding.yaml`, `benchmark/reporter.py`

### ✅ 2.2 Harder coding tasks to escape saturation
**Fixed:** Added three novel-combination tasks that resist memorisation:
- `code_013`: Thread-safe LRU cache (threading.Lock + OrderedDict) — code_exec
- `code_014`: Recursive descent expression parser — no eval(), proper precedence — code_exec
- `code_015`: Consistent hashing ring with virtual nodes (hashlib, add/remove/get) — code_exec  
`asyncio`-based rate limiter deferred (timing-dependent tests are flaky in code_exec).  
**File:** `tasks/coding.yaml`

### ✅ 2.3 Post-training-cutoff knowledge tasks
**Fixed:** Added 3 tasks for January–February 2025 events:
- `post_cutoff_president_47`: 47th US President — Donald Trump (January 2025)
- `post_cutoff_deepseek_r1`: Company that released DeepSeek-R1 in January 2025
- `post_cutoff_super_bowl_lix`: Super Bowl LIX winner — Philadelphia Eagles (February 2025)  
**File:** `tasks/knowledge.yaml`

### ✅ 2.4 Multi-step planning agentic task
**Fixed:** Added `plan_001` — plan a Python CLI scraper → SQLite → --query in ≤12 steps,
scored via `rubric_judge` on step count, concreteness, stdlib constraint, db naming, and
case-insensitive --query implementation.  
**File:** `tasks/agentic.yaml`

---

## Tier 3 — Infrastructure

### ✅ 3.1 `multi_contains` scorer
**Fixed:** Added `_score_multi_contains` to `benchmark/scorer.py`. Supports
`groups` (OR within group, AND across groups) and `values` (flat AND list).
Registered in the dispatch table.

### ✅ 3.2 Contamination-adjusted score in reporter
**Fixed:** Added "Excl. memorised" row to the accuracy table that filters out
tasks with `contamination_risk: high`. Completed together with 2.1.  
**File:** `benchmark/reporter.py`

### ✅ 3.3 Benchmark saturation warning
**Fixed:** After the difficulty breakdown, prints a dim yellow warning when any
model exceeds 85% mean score: "⚠ Scores near ceiling — consider adding harder tasks."  
**File:** `benchmark/reporter.py`

### ✅ 3.4 Judge enablement via CLI flag
**Fixed:** Added `--judge-model MODEL`, `--judge-api-key KEY`, and
`--judge-base-url URL` flags to `run.py`. Local discovered models are wired
up automatically; external models use the URL + key pair. The interactive TTY
prompt is skipped when `--judge-model` is supplied. Added `import os` (was
missing — the interactive path already used `os.environ`).  
**File:** `run.py`

---

## Tier 4 — Longer term

### ✅ 4.1 Task freshness / rotating pool
**Lesson from SWE-bench Pro:** Fixed task sets become contaminated over time
as they appear in fine-tuning datasets and blog posts. SWE-bench Pro rotates
~300 problems to prevent memorisation.  
**Fixed:** Tasks carry an `introduced: YYYY-MM-DD` field (tagged the
2026-04-27 batch — code_013–015, plan_001, post_cutoff_* — and the
2026-06-04 batch — code_016–018, reason_009–011 — from git history).
`--exclude-before DATE` runs only tasks introduced on/after the cutoff;
untagged legacy tasks are excluded so the subset stays contamination-resistant.
Tag every new task going forward.  
**Files:** `benchmark/loader.py` (`filter_introduced_since`), `benchmark/cli.py`,
`tasks/coding.yaml`, `tasks/reasoning.yaml`, `tasks/knowledge.yaml`, `tasks/agentic.yaml`

### ✅ 4.2 Harness-controlled thinking mode A/B
**Lesson from SWE-bench scaffolding finding:** The same model scored 69% vs
81% depending on whether an agent harness was used — a 12-point
infrastructure gap. Our thinking models get internal "scaffolding" from
chain-of-thought but we don't isolate it.  
**Fixed:** `--ab-thinking` runs every task twice per model — `[think]` and
`[no-think]` arms appear as sibling columns in every existing table, plus a
dedicated "Thinking A/B" delta table (Δ score, think tokens, Δ latency).
An explicit task-level `thinking` value now wins over config in the Ollama
backend and `think=false` is sent on the wire (thinking models default to
on). Backends declare `supports_thinking_ab`; unsupported backends run a
single arm with a warning. Arms can't collide in the resume cache — the
label and the task fingerprint both differ.  
*Validated offline against the fake-client harness; pending one live Ollama
run to confirm wire behavior.*  
**Files:** `benchmark/cli.py`, `benchmark/session.py`,
`benchmark/backends/base.py`, `benchmark/backends/ollama.py`,
`benchmark/reporter.py`, `tests/test_ab_thinking.py`

### ✅ 4.3 Contamination audit mode
**Lesson from SWE-bench:** Frontier models could reproduce verbatim gold
patches from task IDs alone. OpenAI caught this by checking chain-of-thought
for knowledge that wasn't in the problem description.  
**Fixed:** `--audit-contamination` probes each model with each code_exec
task's *id only* (the probe never contains the problem statement — tested).
The audit extracts the solution's identifying signals (function names the
test harness calls, builtins excluded) and flags a task when the model
reproduces ≥50% of them from the opaque id. Reports a per-model table
(flagged count, claimed-unknown count, flagged task ids) and saves
`audit_<ts>.json`. Probe-only mode — no scoring run.
*Validated offline against a fake runner; pending a live model run.*  
**Files:** new `benchmark/auditor.py`, `benchmark/cli.py`, `tests/test_auditor.py`
