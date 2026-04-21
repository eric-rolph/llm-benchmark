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
| `category` | string | One of: `math`, `reasoning`, `coding`, `instruction_following`, `knowledge`, `writing`, `summarization` |
| `prompt`   | string | The question or instruction sent to the model |
| `scoring`  | dict   | Scoring configuration (see below) |

## Optional Fields

| Field         | Type    | Default | Description |
|---------------|---------|---------|-------------|
| `system`      | string  | none    | System prompt prepended before user turn |
| `temperature` | float   | from config | Override per-task temperature |
| `max_tokens`  | int     | from config | Override per-task token limit |
| `thinking`    | boolean | false   | Request thinking/reasoning mode (Ollama only) |

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
Parses the first JSON object found in the response and checks for required keys.

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

### `llm_judge`
Placeholder for LLM-as-judge scoring. Requires `--judge` flag (not yet implemented).
Currently returns `None` score marked for manual review.

---

## Conventions

- `id` values must be unique across all task files — they're used as keys in results JSON
- Use `value:` for the expected answer key in new tasks (`answer:` is accepted for backwards compatibility)
- For `code_exec` tasks, structure your prompt so models return a single function in a fenced code block
- Categories must match a `tasks/*.yaml` stem for `--category` filtering to work
- Add new categories by creating a new `tasks/mycategory.yaml` file — they are auto-discovered
