# LLM Benchmark Suite

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tasks](https://img.shields.io/badge/tasks-78-green)
![Backends](https://img.shields.io/badge/backends-3-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Automated, objective benchmarking for **local LLMs** — no cloud, no API keys, no opinion.  
Supports **LM Studio**, **Ollama**, and **llama.cpp** with automatic model discovery.  
Cross-platform: Windows · macOS · Linux.

---

## Why This Exists

Most LLM leaderboards measure proprietary models on curated benchmarks you can't reproduce.
This tool runs **deterministic, open tasks** against models you already have running on your own machine.
Results are reproducible: temperature=0, tolerance-based numeric scoring, strict schema validation.

---

## Features

| Feature | Details |
|---|---|
| **Multi-backend** | LM Studio, Ollama, llama.cpp — configure once, run against all |
| **Auto-discovery** | Probes enabled backends, enumerates all available models |
| **Thinking model support** | Qwen3, DeepSeek-R1 — reasoning tokens captured, clean text scored |
| **78 tasks, 7 categories** | Math, reasoning, coding, knowledge, writing, summarization, instruction-following |
| **14 scoring types** | numeric, exact, contains, fuzzy_match, regex, json_keys, line_count, code_exec, word_count, contains_n, not_contains, ends_with, pass_at_k, llm_judge |
| **Few-shot examples** | Add `few_shot:` to any task YAML to inject conversation history before the prompt |
| **pass@k coding** | `scoring.type: pass_at_k` runs a task k times; uses the unbiased Chen et al. (2021) estimator |
| **LLM-as-judge** | CoT-then-score protocol — enable with `judge.enabled: true` in config |
| **Run resumption** | `--resume` continues from an interrupted run, skipping already-scored tasks |
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
```

**macOS / Linux:**
```bash
cd ~/path/to/llm-benchmark
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate
```

Or install as a package (exposes `llm-bench` command):
```bash
pip install -e .
llm-bench --discover
```

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
   python run.py --discover
   ```

4. **Run everything:**
   ```
   python run.py
   ```

---

## All Commands

```
python run.py                           # auto-discover + run all 78 tasks
python run.py --discover                # probe backends, list models, exit
python run.py --dry-run                 # validate task files + check backends, no inference
python run.py --model "qwen3:8b"        # single model (all categories)
python run.py --backend ollama          # restrict to one backend type
python run.py --category math           # single category
python run.py --task capital_france     # single task by ID
python run.py --limit 2                 # first 2 tasks per category (quick smoke test)
python run.py --resume                  # skip tasks already in the most recent results JSONL
python run.py --no-autoload             # skip LM Studio model-load attempt
python run.py --allow-code-exec         # enable code_exec scoring (runs generated Python locally)
python run.py --ci-threshold 0.80       # exit 1 if overall score < 80%  (CI integration)
python run.py --output my_results       # custom output directory
```

---

## Task Categories

| Category | Tasks | Scoring Method |
|---|---|---|
| `math` | 10 | `numeric` — extract first number, tolerance-based comparison |
| `knowledge` | 15 | `exact` / `contains` / `numeric` |
| `coding` | 12 | `code_exec` / `pass_at_k` — runs generated Python, looks for `PASS` in stdout |
| `reasoning` | 8 | `contains` / `exact` |
| `writing` | 10 | `line_count` / `regex` |
| `summarization` | 8 | `contains` / `line_count` / `regex` |
| `instruction_following` | 15 | `exact` / `contains` / `word_count` / `contains_n` / `not_contains` / `ends_with` |
| **Total** | **78** | |

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
| `pass_at_k` | Any of `k` independent attempts passes `inner_type` scoring (needs `--allow-code-exec`) |
| `llm_judge` | Judge LLM scores the response and returns `SCORE: N` (needs `judge.enabled: true`) |

---

## Results

Results are saved to the `results/` directory after every run:

```
results/
  summary_20250118_143022.json   # full structured results
  summary_20250118_143022.csv    # spreadsheet-friendly export
  results_20250118_143022.jsonl  # incremental crash-safe log (one line per task)
```

Example console output:
```
 ✓  math             arithmetic_addition          score=1.0  1823 t/s  TTFT 42ms
 ~  math             percent_change               score=0.0   Got 12.5, expected 15.0 ±0
 ✓  coding           fizzbuzz_function            score=1.0  1651 t/s
 ✗  reasoning        syllogism_barbara            score=0.0  Missing 'all mortals'
```

---

## Backend Comparison

| Backend | Discovery endpoint | Load model | Thinking tokens |
|---|---|---|---|
| **LM Studio** | `GET /v1/models` | `POST /api/v0/models/load` | `delta.reasoning_content` |
| **Ollama** | `GET /api/tags` | _(auto on first request)_ | `delta.thinking` |
| **llama.cpp** | `GET /health` + `/v1/models` | _(single model, manual start)_ | `<think>` tag stripping |

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

The `task_version` field appears in every JSONL result record for audit trails.

### pass@k scoring

Run a task k times and pass if any attempt succeeds:

```yaml
- id: code_challenge
  category: coding
  prompt: "Write a function called `solve(n)` ..."
  scoring:
    type: pass_at_k
    k: 3
    inner_type: code_exec
    test_code: |
      assert solve(5) == 42
      print('PASS')
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
python run.py --resume
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

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add tasks, backends, and scoring types.

---

## License

MIT


## Supported Backends

| Backend    | Discovery Endpoint | Chat Endpoint | Thinking |
|------------|-------------------|---------------|----------|
| LM Studio  | `GET /v1/models`  | `POST /v1/chat/completions` | `delta.reasoning_content` |
| Ollama     | `GET /api/tags`   | `POST /v1/chat/completions` | `<think>` tags / `think: true` |
| llama.cpp  | `GET /v1/models`  | `POST /v1/chat/completions` | `<think>` tags in content |

## config.yaml

```yaml
backends:
  lm_studio:
    enabled: true
    base_url: "http://localhost:1234/v1"
    auto_discover: true

  ollama:
    enabled: false             # set true when Ollama is running
    base_url: "http://localhost:11434"
    auto_discover: true

  llamacpp:
    enabled: false
    base_url: "http://localhost:8080"
    auto_discover: true

models: []                     # empty = all discovered; or list specific IDs to filter

benchmark:
  temperature: 0.0
  max_tokens: 4096
  timeout: 180
  runs_per_task: 1
```

## Results

Results are saved to `results/results_TIMESTAMP.json` and `.csv` after each run.  
The JSON format includes per-task scores, TPS, TTFT, reasoning token counts, and backend info.


## Benchmark design

| Category | Tasks | Scoring method | What it tests |
|---|---|---|---|
| `math` | 10 | Numeric comparison | Arithmetic, algebra, geometry, sequences |
| `reasoning` | 8 | Numeric / contains | Logic, deduction, probability, patterns |
| `coding` | 8 | **Code execution** | Python correctness via automated unit tests |
| `instruction_following` | 8 | Exact / contains / JSON / line-count | Format compliance, precision |

### Scoring types
- **numeric** — first number extracted from response, compared with optional tolerance  
- **exact** — stripped, case-insensitive equality  
- **contains** — case-insensitive substring check  
- **json_keys** — parses JSON from response, verifies required keys exist  
- **line_count** — counts non-empty lines  
- **code_exec** — extracts code block, runs it with `subprocess`, looks for `PASS` in stdout  

### What gets measured per task
- **Score** (0.0 / 0.5 / 1.0)  
- **TTFT** — time to first token (ms)  
- **TPS** — tokens per second (generation speed)  
- **Total latency** (ms)

## Output

Results are saved to `results/results_TIMESTAMP.{json,csv}` after every run.  
The CSV is ready for Excel / Google Sheets comparison pivot tables.

## Config reference (`config.yaml`)

```yaml
lm_studio:
  base_url: "http://localhost:1234/v1"
  api_key:  "lm-studio"          # any non-empty string works

models:
  - "exact-model-id-from-lm-studio"

benchmark:
  temperature:   0.0    # keep 0.0 for reproducibility
  max_tokens:    512
  timeout:       120    # seconds
  runs_per_task: 1      # increase to 3 for averaged TPS numbers
```

## Adding tasks

Create or edit any `.yaml` file in `tasks/`. Each task needs:

```yaml
tasks:
  - id: my_task_001
    category: math            # math | reasoning | coding | instruction_following
    description: "..."
    prompt: "Your prompt here"
    scoring:
      type: numeric           # see scoring types above
      answer: 42
      tolerance: 0
```

For `code_exec` tasks, add a `test_code` block with assertions that print `PASS` on success:

```yaml
    scoring:
      type: code_exec
      test_code: |
        assert my_function(5) == 120
        print("PASS")
```

## Workflow for comparing Qwen3-6B vs Kimi K2

```powershell
# 1. Load Qwen3-6B in LM Studio, get its ID
python run.py --list-models

# 2. Edit config.yaml, add both model IDs
# 3. Run with Qwen loaded first (auto-load will handle switching if LM Studio supports it)
python run.py

# 4. Results saved to results/ — open the CSV in Excel for comparison
```

> **Note on auto-loading**: LM Studio 0.3.x+ supports programmatic model loading via  
> `POST /api/v0/models/load`. Older versions require manually switching models between runs.
> Use `--no-autoload` if auto-load causes issues.
