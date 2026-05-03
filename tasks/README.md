# Task YAML Schema Reference

Tasks are defined in `.yaml` files in this directory.  
Both formats are supported:

**Format A — bare list** (used by the newer task files):
```yaml
- id: capital_france
  category: knowledge
  prompt: "What is the capital of France?"
  scoring:
    type: exact
    value: Paris
```

**Format B — wrapped dict** (used by the original task files):
```yaml
tasks:
  - id: math_add
    category: math
    prompt: "What is 247 + 389?"
    scoring:
      type: numeric
      answer: 636
      tolerance: 0
```

---

## Required Fields

| Field      | Type   | Description |
|------------|--------|-------------|
| `id`       | string | Unique identifier — no spaces, use `snake_case` |
| `category` | string | One of: `math`, `reasoning`, `coding`, `agentic`, `instruction_following`, `knowledge`, `writing`, `summarization` |
| `prompt`   | string | The question or instruction sent to the model |
| `scoring`  | dict   | Scoring configuration (see below) |

Dataset-driven tasks use `dataset` + `template` instead of `prompt`; see [Dataset Tasks](#dataset-tasks).

## Optional Fields

| Field         | Type    | Default | Description |
|---------------|---------|---------|-------------|
| `system`      | string  | none    | System prompt prepended before user turn |
| `temperature` | float   | from config | Override per-task temperature |
| `max_tokens`  | int     | from config | Override per-task token limit |
| `pass_threshold` | float | 0.8 | Score needed for PASS/FAIL reporting |
| `execution_surface` | string | none | Surface tag for Claw-style reporting, e.g. `local_workspace_repair` |
| `source_signal` | string | none | Demand signal or workflow family that motivated the task |
| `thinking`    | boolean | false   | Request thinking/reasoning mode (Ollama only) |
| `image_url`   | string/list | none | Remote image URL(s) for vision-language tasks |
| `image_path`  | string/list | none | Local image path(s) for vision-language tasks |

---

## Scoring Types

### `numeric`
Extracts the first number from the response and compares with optional tolerance.

```yaml
scoring:
  type: numeric
  value: 42          # expected numeric answer (use 'value' or legacy 'answer')
  tolerance: 0       # allowed absolute error (default 0)
```

### `exact`
Case-insensitive exact match of the trimmed response.

```yaml
scoring:
  type: exact
  value: "Paris"
```

### `contains`
Checks if the expected string appears anywhere in the response (case-insensitive).

```yaml
scoring:
  type: contains
  value: "photosynthesis"
```

### `regex`
Applies a Python `re.search` regex to the response (IGNORECASE | DOTALL).

```yaml
scoring:
  type: regex
  pattern: "\\b[A-Z][a-z]+\\s[A-Z][a-z]+\\b"   # matches a proper name
```

### `json_keys`
Scans the response for valid JSON objects and passes when one contains all required keys.

```yaml
scoring:
  type: json_keys
  keys: [name, age, email]
```

### `line_count`
Counts non-empty lines in the response and checks against expected count.

```yaml
scoring:
  type: line_count
  count: 5
```

### `code_exec`
Extracts a Python code block from the response, appends `test_code`, runs it, and
checks for `PASS` in stdout.

> ⚠️ **Requires `--allow-code-exec` flag.** Code runs with your Python interpreter.
> Review task YAML before enabling. Do not run untrusted task files with this flag.

```yaml
scoring:
  type: code_exec
  test_code: |
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    print("PASS")
```

The model is expected to return a code block containing a function named (e.g.) `add`.
The `test_code` is appended to the extracted code and the full script is executed.

### `logprob_choice`
For multiple-choice/base-model tasks, asks the backend for one token with logprobs
and compares the highest-probability token with the expected answer.

```yaml
scoring:
  type: logprob_choice
  value: "B"
```

### `pass_at_k`
Runs multiple attempts and estimates pass@k using an inner scorer.

```yaml
scoring:
  type: pass_at_k
  k: 3
  n: 5          # optional, alias: samples
  inner_type: code_exec
  test_code: |
    assert solve(5) == 42
    print("PASS")
```

### `workflow_trace`
Scores a model-produced JSON workflow trace with deterministic checks. This is
useful for agentic tasks where the benchmark should grade the path taken and
the final mock state, not just the final prose answer.

```yaml
execution_surface: local_workspace_repair
source_signal: workspace-repair
scoring:
  type: workflow_trace
  min_calls: 3
  required_tools: [read_file, edit_file, run_tests]
  ordered_tools: [read_file, edit_file, run_tests]
  expected_state:
    repo.tests_passed: true
    repo.changed_files:
      contains:
        - benchmark/scorer.py
        - tests/test_scorer.py
  state_contains:
    repo.summary: workflow trace
```

Expected response shape:

```json
{
  "tool_calls": [{"tool": "read_file"}, {"tool": "edit_file"}, {"tool": "run_tests"}],
  "state": {"repo": {"tests_passed": true}}
}
```

`expected_state` uses dotted paths under `state`/`final_state`. Values are exact
by default; use `{contains: ...}` for string/list containment.

### `llm_judge`
Scores subjective tasks with a configured judge model. Enable `judge.enabled: true`
in `config.yaml`; the judge response must include a `SCORE: N` line.

```yaml
scoring:
  type: llm_judge
  criteria: "Does the response answer the prompt accurately and concisely?"
```

### `rubric_judge`
Scores subjective tasks against weighted criteria with a configured judge model.

```yaml
scoring:
  type: rubric_judge
  criteria:
    - criterion: "Directly answers the question"
      weight: 2
    - criterion: "Uses accurate facts"
      weight: 3
```

---

## Dataset Tasks

Dataset-driven tasks use `dataset` + `template` instead of `prompt`. They are
expanded only for real benchmark runs; dry-run schema validation does not load
remote datasets.

```yaml
- id: mmlu_anatomy
  category: knowledge
  dataset:
    name: "cais/mmlu"
    subset: "anatomy"
    split: "test"
    limit: 100
  template: |
    Question: {{ question }}
    Answer with just the letter.
  scoring:
    type: exact
    answer_field: answer
```

`trust_remote_code` defaults to `false`. Set it to `true` inside `dataset:` only
for datasets you explicitly trust.

---

## Conventions

- `id` values must be unique across all task files — they're used as keys in results JSON
- Use `value:` for the expected answer key in new tasks (`answer:` is accepted for backwards compatibility)
- For `code_exec` tasks, structure your prompt so models return a single function in a fenced code block
- Categories must match a `tasks/*.yaml` stem for `--category` filtering to work
- Add new categories by creating a new `tasks/mycategory.yaml` file — they are auto-discovered
