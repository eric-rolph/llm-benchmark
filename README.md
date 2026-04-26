# LLM Benchmark Suite

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tasks](https://img.shields.io/badge/tasks-87-green)
![Backends](https://img.shields.io/badge/backends-9-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Local-first, reproducible benchmarking for LLMs you run yourself.

`llm-benchmark` runs deterministic task suites against local and OpenAI-compatible inference servers, then writes comparison-friendly JSON, JSONL, CSV, and optional HTML reports. It supports **LM Studio, Ollama, llama.cpp, vLLM, SGLang, TensorRT-LLM, TGI, KTransformers, and generic OpenAI-compatible backends** with automatic model discovery.

Current state: **87 tasks**, **7 scored categories**, **16 scoring modes**, crash-safe resume, pass@k, LLM/rubric judging, result diffs, and pairwise arena mode with persisted ELO artifacts.

---

## Why This Exists

Most LLM leaderboards measure proprietary models on curated benchmarks you can't reproduce.
This tool runs **deterministic, open tasks** against models you already have running on your own machine or on an OpenAI-compatible server you control.
Results are built for auditability: temperature=0 by default, tolerance-based numeric scoring, strict schema validation, task hashes, versioned JSONL records, and run-to-run comparison.

---

## At a Glance

| Area | Current state |
|---|---|
| **Tasks** | 87 tasks across math, knowledge, coding, reasoning, writing, summarization, and instruction-following |
| **Backends** | LM Studio, Ollama, llama.cpp, vLLM, SGLang, TensorRT-LLM, TGI, KTransformers, generic OpenAI-compatible |
| **Scoring** | 16 scoring modes, including exact/numeric/regex/JSON checks, code execution, pass@k, logprob choice, LLM judge, and rubric judge |
| **Reproducibility** | Task version/hash tracking, opt-in Hugging Face auto-config, dataset dry-run safety, resumable JSONL logs |
| **Outputs** | Rich console tables, JSON, CSV, crash-safe JSONL, optional HTML reports, result comparisons, arena ELO JSON |

---

## Features

| Feature | Details |
|---|---|
| **Multi-backend** | LM Studio, Ollama, llama.cpp, vLLM, TGI, SGLang, TensorRT, KTransformers |
| **Auto-discovery** | Probes enabled backends, enumerates all available models |
| **HF Auto-Config** | Opt-in fetch of Hugging Face `generation_config.json` parameters for tested models |
| **Thinking model support** | Qwen3, DeepSeek-R1 — reasoning tokens captured, clean text scored |
| **87 tasks, 7 categories** | Math, reasoning, coding, knowledge, writing, summarization, instruction-following, including vision-language tasks folded into reasoning/writing |
| **16 scoring types** | numeric, exact, contains, fuzzy_match, regex, json_keys, line_count, code_exec, word_count, contains_n, not_contains, ends_with, logprob_choice, pass_at_k, llm_judge, rubric_judge |
| **Few-shot examples** | Add `few_shot:` to any task YAML to inject conversation history before the prompt |
| **pass@k coding** | `scoring.type: pass_at_k` can run n samples and estimate pass@k with the unbiased Chen et al. (2021) estimator |
| **LLM-as-judge** | CoT-then-score protocol — enable with `judge.enabled: true` in config |
| **Run resumption** | `--resume` continues from an interrupted run, skipping task-version/content-matched results |
| **Result comparison** | `--compare` diffs saved JSON/JSONL runs, including model-level and task-level score movement |
| **Arena artifacts** | `--arena` runs pairwise ELO judging and persists leaderboard + match history JSON |
| **Task versioning** | `metadata.version` in task YAML propagates to JSONL for audit trails |
| **Composite score** | Weighted cross-category score in summary table (coding/math/reasoning weighted higher) |
| **Crash-safe results** | Incremental JSONL written after every task — restart safely |
| **Latency histograms** | Per-category min/median/p95/max latency surfaced in summary table |
| **CI integration** | `--ci-threshold` flag returns exit code 1 when score drops below target |
| **Rich console output** | Live per-task scores + summary tables + CSV/JSON export |

---

## Requirements

- Python 3.11+
- At least one inference backend running locally (LM Studio / Ollama / llama.cpp)

---

## Setup

**Windows:**
```powershell
cd C:\path\to\llm-benchmark
.\setup.ps1
.\.venv\Scripts\Activate.ps1
llm-bench --dry-run
```

**macOS / Linux:**
```bash
cd ~/path/to/llm-benchmark
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate
llm-bench --dry-run
```

Manual editable install, if you created the virtual environment yourself:
```bash
pip install -e .
llm-bench --discover
```

On PowerShell, `run.py` by itself will not execute from the current directory. Prefer the installed `llm-bench` command. If you want to run the script file directly, use `python .\run.py --dry-run`.

---

## Quick Start

1. **Start your backend:**
   - **LM Studio** — open app → enable Local Server (Settings → Local Server)
   - **Ollama** — `ollama serve` (starts automatically on most installs)
   - **llama.cpp** — `./llama-server -m model.gguf --port 8080`

2. **Enable the backend in `config.yaml`:**
   ```yaml
   backends:
     lm_studio:
       enabled: true
       base_url: "http://localhost:1234"
       api_key: "lm-studio"          # placeholder — LM Studio ignores this
     ollama:
       enabled: false
       base_url: "http://localhost:11434"
     llamacpp:
       enabled: false
       base_url: "http://localhost:8080"
   ```
   > **Security note:** API keys in `config.yaml` are only placeholders for local servers.
   > For real keys, use environment variables: `LLM_BENCH_<BACKEND_NAME>_API_KEY`.

3. **Discover available models:**
   ```
   llm-bench --discover
   ```

4. **Run everything:**
   ```
   llm-bench
   ```

---

## All Commands

```
llm-bench                           # auto-discover + run all 87 tasks
llm-bench --discover                # probe backends, list models, exit
llm-bench --dry-run                 # validate task files + check backends, no inference
llm-bench --model "qwen3:8b"        # single model (all categories)
llm-bench --backend ollama          # restrict to one backend type
llm-bench --category math           # single category
llm-bench --task capital_france     # single task by ID
llm-bench --limit 2                 # first 2 tasks per category (quick smoke test)
llm-bench --resume                  # skip tasks already in the most recent results JSONL
llm-bench --compare old.jsonl new.jsonl # compare two saved result files
llm-bench --compare old.json new.json --compare-top 20 # show more task deltas
llm-bench --no-autoload             # skip LM Studio model-load attempt
llm-bench --allow-code-exec         # enable code_exec scoring (runs generated Python locally)
llm-bench --ci-threshold 0.80       # exit 1 if overall score < 80%  (CI integration)
llm-bench --html-report             # write an interactive HTML report
llm-bench --arena                   # pairwise ELO arena using an LLM judge
llm-bench --output my_results       # custom output directory
```

Script fallback from the repo root: replace `llm-bench` with `python .\run.py` on Windows PowerShell or `python run.py` on macOS/Linux.

### Windows troubleshooting

If PowerShell says `run.py` exists but is not recognized, that is normal PowerShell command precedence. Use `llm-bench --dry-run` or `python .\run.py --dry-run`.

If `python .\run.py --dry-run` fails with `ModuleNotFoundError: No module named 'yaml'`, your active `python` is not the repo environment or the package was not installed. From the repo root:

```powershell
deactivate  # ignore if no environment is active
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip install -e .
llm-bench --dry-run
```

---

## Task Categories

| Category | Tasks | Scoring Method |
|---|---|---|
| `math` | 10 | `numeric` — extract first number, tolerance-based comparison |
| `knowledge` | 15 | `exact` / `contains` / `numeric` |
| `coding` | 12 | `code_exec` / `pass_at_k` — runs generated Python, looks for `PASS` in stdout |
| `reasoning` | 12 | `contains` / `exact` / `fuzzy_match` / `word_count` / `llm_judge` |
| `writing` | 13 | `line_count` / `regex` / `word_count` |
| `summarization` | 10 | `contains` / `line_count` / `regex` |
| `instruction_following` | 15 | `exact` / `contains` / `word_count` / `contains_n` / `not_contains` / `ends_with` |
| **Total** | **87** | |

---

## Scoring Types Reference

| Type | Passes when |
|---|---|
| `numeric` | Extracted number is within `tolerance` of `answer`/`value` |
| `exact` | Stripped, lowercased response equals expected |
| `contains` | Response contains expected substring (case-insensitive) |
| `fuzzy_match` | Bidirectional: `answer ⊆ response` OR `response ⊆ answer` — handles valid paraphrases |
| `contains_n` | Expected substring appears at least `min_count` times |
| `not_contains` | None of the `forbidden` strings appear in the response |
| `ends_with` | Last non-empty line's final word matches `answer` |
| `word_count` | Response word count falls within `[min, max]` |
| `regex` | Response matches `pattern` |
| `json_keys` | Response parses as JSON and contains all `keys` |
| `line_count` | Non-empty line count equals `count` |
| `code_exec` | Generated code block executes and prints `PASS` (needs `--allow-code-exec`) |
| `logprob_choice` | Highest-probability one-token choice matches `answer`/`value` |
| `pass_at_k` | Estimates pass@k over `n`/`samples` independent attempts using `inner_type` scoring |
| `llm_judge` | Judge LLM scores the response and returns `SCORE: N` (needs `judge.enabled: true`) |
| `rubric_judge` | Judge LLM scores weighted criteria and returns `RUBRIC_SCORE: N` |

---

## Results

Results are saved to the `results/` directory after every run:

```
results/
  results_20250118_143022.json   # full structured results
  results_20250118_143022.csv    # spreadsheet-friendly export
  results_20250118_143022.jsonl  # incremental crash-safe log (one line per task)
  arena_20250118_143022.json     # arena leaderboard + match history when --arena is used
```

Example console output:
```
 ✓  math             arithmetic_addition          score=1.0  1823 t/s  TTFT 42ms
 ~  math             percent_change               score=0.0   Got 12.5, expected 15.0 ±0
 ✓  coding           fizzbuzz_function            score=1.0  1651 t/s
 ✗  reasoning        syllogism_barbara            score=0.0  Missing 'all mortals'
```

---

### Compare result files

Compare two completed or in-progress runs without starting a backend:

```bash
llm-bench --compare results/results_20250118_143022.jsonl results/results_20250119_091500.jsonl
```

`--compare` accepts the structured JSON files and the incremental JSONL files. It reports per-model score movement, composite-score movement, unmatched task counts, and the largest task-level deltas. Use `--compare-top N` to change how many task changes are shown.

---

## Backend Comparison

| Backend | Discovery endpoint | Chat Endpoint | Thinking tokens |
|---|---|---|---|
| **LM Studio** | `GET /v1/models` | `POST /v1/chat/completions` | `delta.reasoning_content` |
| **Ollama** | `GET /api/tags` | `POST /v1/chat/completions` | `<think>` tags / `think: true` |
| **llama.cpp** | `GET /v1/models` | `POST /v1/chat/completions` | `<think>` tag stripping |
| **vLLM** | `GET /v1/models` | `POST /v1/chat/completions` | OpenAI compatible |
| **TGI** | `GET /v1/models` | `POST /v1/chat/completions` | OpenAI compatible |
| **SGLang** | `GET /v1/models` | `POST /v1/chat/completions` | OpenAI compatible |
| **TensorRT-LLM** | `GET /v1/models` | `POST /v1/chat/completions` | OpenAI compatible |
| **KTransformers** | `GET /v1/models` | `POST /v1/chat/completions` | OpenAI compatible |

---

## Adding Tasks

Tasks are YAML files in `tasks/`. See [tasks/README.md](tasks/README.md) for the full schema.

Quick example:
```yaml
- id: capital_germany
  category: knowledge
  prompt: "What is the capital of Germany? Answer with one word."
  scoring:
    type: exact
    answer: berlin
```

### Few-shot examples

Add a `few_shot:` block to any task to inject conversation history before the model sees the actual prompt:

```yaml
- id: my_task
  category: coding
  prompt: "Write a Python function called `square(n)` that returns n squared."
  few_shot:
    - user: "Write a Python function called `cube(n)` that returns n cubed."
      assistant: |
        def cube(n):
            return n ** 3
  scoring:
    type: code_exec
    test_code: |
      assert square(3) == 9
      print('PASS')
```

### Task versioning

Add a `metadata:` block to any dict-format task file to version-stamp all its tasks:

```yaml
metadata:
  version: 2

tasks:
  - id: my_task
    ...
```

The `task_version` and `task_hash` fields appear in every JSONL result record for audit trails. `--resume` only reuses a cached row when the model id, task id, task version, and task hash all match the current task definition.

### pass@k scoring

Run a task multiple times and estimate pass@k:

```yaml
- id: code_challenge
  category: coding
  prompt: "Write a function called `solve(n)` ..."
  scoring:
    type: pass_at_k
    k: 3        # report pass@3
    n: 5        # optional: draw 5 samples for a better estimator (alias: samples)
    inner_type: code_exec
    test_code: |
      assert solve(5) == 42
      print('PASS')
```

If `n`/`samples` is omitted, the runner uses `benchmark.runs_per_task`, but never fewer than `k`.

### HF auto-config

By default, runs do not contact Hugging Face. To opt into fetching `generation_config.json` / `config.json` for model IDs that contain `/`, enable:

```yaml
benchmark:
  hf_auto_config: true
```

Fetched settings are recorded in result JSON/JSONL as `hf_generation_config`.

### Dataset-driven tasks

Dataset tasks can expand Hugging Face datasets into concrete prompts when the optional `datasets` extra is installed. Simple `--dry-run` validation does not expand datasets, avoiding network side effects during schema checks.

Remote dataset code is not trusted by default. If a dataset truly requires it, opt in per task:

```yaml
dataset:
  name: "example/dataset"
  trust_remote_code: true
```

### LLM-as-judge

For subjective tasks, enable the LLM judge in `config.yaml`:

```yaml
judge:
  enabled: true
  model: null   # null = use first discovered model
```

Then add tasks using `llm_judge` scoring:

```yaml
- id: explain_quantum
  category: writing
  prompt: "Explain quantum entanglement in two sentences for a teenager."
  scoring:
    type: llm_judge
    criteria: "Is the explanation accurate, concise, and appropriate for a teenager?"
    reference: "Quantum entanglement means two particles stay linked so measuring one instantly affects the other, no matter how far apart they are."
```

The judge uses a CoT-then-score protocol: it reasons step by step and then outputs `SCORE: N` (0–10), which is normalised to 0.0–1.0.

### Run resumption

Resume an interrupted benchmark without re-running completed tasks:

```bash
llm-bench --resume
```

Or enable it permanently in `config.yaml`:

```yaml
benchmark:
  resume: true
```

### Composite score

The summary table now includes a **Composite ★** row — a weighted average across categories where harder categories carry more weight:

| Category | Weight |
|---|---|
| coding | 1.5 |
| math | 1.2 |
| reasoning | 1.2 |
| knowledge | 1.0 |
| instruction_following | 1.0 |
| summarization | 0.8 |
| writing | 0.8 |

---

## Changelog

### Accuracy improvements (agent review)

| Area | Change |
|---|---|
| **TTFT (reasoning models)** | Previously used the first *content* token; now uses `min(t_first_content, t_first_reasoning)` so models that emit reasoning tokens before content tokens report correct TTFT |
| **TPS accuracy** | Returns `None` instead of a word-count proxy when the API does not report `completion_tokens`; prevents misleading cross-model comparisons |
| **Reasoning token counting** | Reads `completion_tokens_details.reasoning_tokens` from the API response instead of counting stream chunks |
| **Multi-run all-errors** | Metrics are `None` (not `0.0`) when every run in a multi-run task fails |
| **pass@k estimator** | Uses the unbiased Chen et al. (2021) estimator instead of a binary any-pass check; allows n > k sampling for better statistical estimates |
| **Empty code block** | `code_exec` returns score 0.0 with an explicit message when no Python block is found, instead of executing an empty string |
| **LLM judge prompt injection** | Response is wrapped in `<response>` XML delimiters and truncated to 4,000 chars; `reference` answer truncated to 2,000 chars |
| **JSONL `model` field** | Now records `model_id` (the actual model name) rather than the backend name |
| **Composite score categories** | Lookup is now case-insensitive, preventing uncategorised tasks from being silently weighted 1.0 |
| **`line_count` key** | Scorer now accepts both `value` and `count` YAML keys (some tasks used `count`) |

### Bug fixes (live-run validation)

| Area | Change |
|---|---|
| **Accent-insensitive `contains`** | `_score_contains` now normalises both needle and haystack with NFKD Unicode decomposition before comparison, so e.g. `"brasilia"` matches `"Brasília"` |
| **Sentence-based writing tasks** | Tasks that ask for "N sentences" now instruct the model to output one sentence per line, making `line_count` scoring reliable |
| **Accent-insensitive `contains_n` / `ends_with` / `fuzzy_match`** | All remaining string-comparison scorers updated to use NFKD normalisation, consistent with `contains`; e.g. `"cafe"` matches `"café"` in all four scorer types |
| **`pass_at_k` n > k sampling** | `score_pass_at_k` now reads `k` from `task["scoring"]["k"]` (falling back to `len(run_results)`), enabling over-sampling (`n > k`) with the Chen et al. 2021 unbiased estimator |
| **Version-safe resume** | `--resume` now hydrates cached rows into reports and only skips results with matching model id, task id, task version, and task hash |
| **HF auto-config opt-in** | Hugging Face generation-config fetches are disabled by default and recorded in result metadata when enabled |
| **Safer dataset expansion** | `--dry-run` validates dataset task schema without expansion; Hugging Face `trust_remote_code` defaults to false |
| **Robust `json_keys` parsing** | Scoring now scans for valid JSON objects instead of using a greedy brace regex |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add tasks, backends, and scoring types.

---

## License

MIT



