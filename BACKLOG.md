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

### 4.1 Task freshness / rotating pool
**Lesson from SWE-bench Pro:** Fixed task sets become contaminated over time
as they appear in fine-tuning datasets and blog posts. SWE-bench Pro rotates
~300 problems to prevent memorisation.  
**Fix (when scale warrants it):** Tag tasks with a `introduced` date. Add a
`--exclude-before DATE` flag that runs only tasks introduced after a given
date, making it easy to create a "fresh" subset as the suite grows.

### 4.2 Harness-controlled thinking mode A/B
**Lesson from SWE-bench scaffolding finding:** The same model scored 69% vs
81% depending on whether an agent harness was used — a 12-point
infrastructure gap. Our thinking models get internal "scaffolding" from
chain-of-thought but we don't isolate it.  
**Fix:** For Ollama (which supports `think=true/false`), run each task twice
when `--ab-thinking` is set and report both scores. This makes the
thinking-vs-no-thinking delta visible per model, similar to what E3 captures
but more direct.  
**Files:** `run.py`, `benchmark/backends/ollama.py`, `benchmark/reporter.py`

### 4.3 Contamination audit mode
**Lesson from SWE-bench:** Frontier models could reproduce verbatim gold
patches from task IDs alone. OpenAI caught this by checking chain-of-thought
for knowledge that wasn't in the problem description.  
**Fix:** Add `--audit-contamination` mode that, for each coding task, sends
the model *only* the task ID (not the problem) and checks if the response
contains the expected function name and structure. A high match rate flags
contamination. Requires judge or regex scoring.  
**Files:** `run.py`, new `benchmark/auditor.py`
