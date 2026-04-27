# LLM Benchmark — Backlog

Prioritised list of improvements derived from benchmark runs, the
akitaonrails coding-benchmark article, the qwopus36-eval review, and
the OpenAI SWE-bench Verified post-mortem.

---

## Tier 1 — Bug fixes (wrong results today)

### 1.1 `headline_from_article` regex is too narrow
**Problem:** Correct headlines from nemotron and gemma-4-31b score 0 because
the multi-part regex `(?i)(battery|charg).{0,60}(stanford|smartphone|fast|rapid|new|speed)`
requires both halves to appear within 60 characters. Models write valid
headlines that don't match the pattern — this is the "test too narrowly
defined" failure mode identified in the SWE-bench post-mortem.  
**Fix:** Replace with two independent `contains` checks (battery/charg AND
stanford/fast/new) joined in a `multi_contains` scorer, or widen the regex
window and drop the second anchor.  
**File:** `tasks/summarization.yaml`

### 1.2 `vision_describe_dog` keyword is too fragile
**Problem:** Some models say "canine" or describe the animal without the
literal word "dog", scoring 0 despite a correct response.  
**Fix:** Change scorer from `contains: dog` to `fuzzy_match` or a short
`regex` alternation `(dog|canine|puppy)`.  
**File:** `tasks/vision.yaml`

### 1.3 Qwen3 thinking-mode code block extraction fails
**Problem:** `code_009`–`code_012` report "no Python code block found" on
both Qwen3 models. Qwen3 in thinking mode wraps its answer differently;
the code ends up outside a fenced block or inside the thinking stream.  
**Fix:** Investigate the raw response for these tasks (check JSONL
`response_preview`). Either improve `_extract_code` to handle the format,
or add a pre-processing step in the Qwen3/LM Studio backend path to
normalise the output before scoring.  
**Files:** `benchmark/scorer.py`, `benchmark/backends/lm_studio.py`

---

## Tier 2 — Task quality

### 2.1 Add `contamination_risk` field to LeetCode-style tasks
**Problem:** Classic tasks (`longest_increasing_subsequence`, `group_anagrams`,
`max_subarray_sum`, `two_sum`, etc.) are in every training corpus. High scores
may reflect recall, not reasoning. The SWE-bench post-mortem showed frontier
models reproducing verbatim solutions from memory.  
**Fix:** Add `contamination_risk: high` to known textbook problems. The
reporter can then print a separate "contamination-adjusted score" that
excludes high-risk tasks, giving a cleaner capability signal.  
**Files:** `tasks/coding.yaml`, `benchmark/reporter.py`

### 2.2 Harder coding tasks to escape saturation
**Problem:** Models are scoring 87–91% overall on the first run. The
SWE-bench post-mortem notes benchmarks near 80% stop measuring capability.
Novel-combination tasks resist memorisation and better differentiate models.  
**Candidates:**
- Thread-safe LRU cache using only stdlib (`threading.Lock` + `OrderedDict`)
- Recursive descent expression parser (tokenise → parse → evaluate)
- Consistent hashing ring with virtual nodes
- `asyncio`-based rate limiter  
**Scoring:** `code_exec` with a hidden test suite, same as existing coding tasks.  
**File:** `tasks/coding.yaml`

### 2.3 Post-training-cutoff knowledge tasks
**Problem:** All current knowledge tasks test facts that have been in training
data for years (capital of France, Carbon-14 half-life, etc.). There is no
signal for whether a model is reasoning vs. reciting.  
**Fix:** Add 3–5 knowledge tasks about events from 2025–2026 that postdate
most training cutoffs. Use `contains` or `numeric` scoring on verifiable
facts (election results, sports records, scientific announcements, etc.).  
**File:** `tasks/knowledge.yaml`

### 2.4 Multi-step planning agentic task
**Problem:** Identified in qwopus36-eval as a meaningful capability signal;
not yet in our suite. Tests whether a model can decompose a goal into a
numbered, executable plan with specific tool calls and constraints.  
**Fix:** Add `plan_001`: "Plan a Python CLI tool that scrapes a URL, stores
results in SQLite, and exposes a `--query` flag — in ≤12 numbered steps
with exact shell commands." Score via `rubric_judge` (completeness,
specificity, executability, constraint adherence).  
**File:** `tasks/agentic.yaml`

---

## Tier 3 — Infrastructure

### 3.1 `multi_contains` scorer
**Problem:** Several tasks need "response must contain A AND B" but the only
options are `contains` (single needle) or `regex` (error-prone for
non-regex authors). The headline fix in 1.1 needs this.  
**Fix:** Add `_score_multi_contains` — takes a `values: [...]` list and
scores 1.0 only when all needles are present (case-insensitive). Register
as `multi_contains` in the dispatch table.  
**File:** `benchmark/scorer.py`

### 3.2 Contamination-adjusted score in reporter
**Depends on:** 2.1  
**Fix:** In `print_report`, add a "Clean Score" row to the accuracy table
that excludes tasks with `contamination_risk: high`. Annotate the column
header so it's clear what's excluded.  
**File:** `benchmark/reporter.py`

### 3.3 Benchmark saturation warning
**Problem:** When overall scores exceed ~85%, the benchmark differentiates
poorly between models — the SWE-bench lesson. We have no signal for this.  
**Fix:** In the summary footer, print a dim warning when the mean score
across all models exceeds 0.85: "⚠ Scores near ceiling — consider adding
harder tasks."  
**File:** `benchmark/reporter.py`

### 3.4 Judge enablement via CLI flag
**Problem:** Currently the only way to enable the LLM judge is interactively
at startup (TTY only) or via `config.yaml`. CI pipelines and scripted runs
that want judge scoring have no clean path.  
**Fix:** Add `--judge-model MODEL` CLI flag that sets `judge_client` and
`judge_model` at parse time, bypassing the interactive prompt. If the model
ID is not in discovered backends, require `--judge-api-key` and
`--judge-base-url` flags.  
**Files:** `run.py`

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
