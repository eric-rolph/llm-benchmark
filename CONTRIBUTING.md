# Contributing to llm-benchmark

Thank you for your interest in contributing!  
This guide covers: adding backends, adding tasks, adding scoring types, and submitting PRs.

---

## Quick Start

```bash
git clone https://github.com/ericrichardson/llm-benchmark.git
cd llm-benchmark
./setup.sh          # macOS/Linux
# or: .\setup.ps1   # Windows

# Run tests
.venv/bin/python -m pytest tests/ -v

# Validate task files
python run.py --dry-run
```

---

## Adding Tasks

Tasks are plain YAML files in `tasks/`. No Python needed.

1. **Create or edit a YAML file** in `tasks/`:
   - Add to an existing file to extend a category
   - Create `tasks/mycategory.yaml` to add a new category (auto-discovered)

2. **Follow the schema** documented in `tasks/README.md`

3. **Test your task** before submitting:
   ```bash
   python run.py --dry-run          # validate task files, no inference
   python run.py --task your_task_id  # run one task against live models
   ```

4. **Example task:**
   ```yaml
   - id: capital_australia
     category: knowledge
     prompt: "What is the capital of Australia?"
     scoring:
       type: exact
       value: Canberra
   ```

### Task authoring guidelines
- `id` values must be globally unique across all task files
- Prompts should be unambiguous — the same prompt should produce consistent answers
- Use `exact` only when the model must return *only* the answer (no extra words)
- Prefer `contains` when the answer might appear in a sentence
- Prefer `numeric` with a non-zero `tolerance` for computed values (avoids rounding failures)
- `code_exec` tasks require `--allow-code-exec` to run — document this in the task comment

---

## Adding a Backend

Backends are self-contained Python classes in `benchmark/backends/`.

1. **Create `benchmark/backends/mybackend.py`** implementing the `Backend` ABC:

   ```python
   from benchmark.backends.base import Backend, ModelInfo
   import requests

   class MyBackend(Backend):
       def is_available(self) -> bool:
           try:
               return requests.get(f"{self._api_root()}/health", timeout=3).ok
           except Exception:
               return False

       def discover_models(self) -> list[ModelInfo]:
           r = requests.get(f"{self._v1_url()}/models", timeout=5)
           return [
               ModelInfo(id=m["id"], name=m["id"], backend_name=self.name)
               for m in r.json().get("data", [])
           ]
   ```

2. **Register it** in `benchmark/backends/__init__.py`:
   ```python
   from benchmark.backends.mybackend import MyBackend
   _REGISTRY["mybackend"] = MyBackend
   ```

3. **Add a config block** to `config.yaml`:
   ```yaml
   backends:
     mybackend:
       enabled: true
       name: "My Backend"
       base_url: "http://localhost:9090"
   ```

4. **Test it:**
   ```bash
   python run.py --backend mybackend --discover
   ```

### Backend interface overview

| Method | Required? | Purpose |
|--------|-----------|---------|
| `is_available()` | ✅ | Returns True if server is reachable |
| `discover_models()` | ✅ | Returns list of `ModelInfo` objects |
| `get_openai_client()` | override if needed | Returns configured `OpenAI` client |
| `get_extra_chat_params(task)` | override if needed | Inject backend-specific params into chat call |
| `ensure_model_loaded(model_id)` | override if needed | Pre-load a model before inference |

---

## Adding a Scoring Type

Scoring types are defined in `benchmark/scorer.py`.

1. **Add your function** following the `(response: str, scoring: dict) -> tuple[float, str]` signature:
   ```python
   def _score_my_type(response: str, scoring: dict) -> tuple:
       # ... scoring logic ...
       return 1.0, "Passed"  # or 0.0, "Reason for failure"
   ```

2. **Register it** in the `dispatch` dict inside `score_response()`:
   ```python
   dispatch = {
       ...
       "my_type": _score_my_type,
   }
   ```

3. **Add a test** in `tests/test_scorer.py`.

4. **Document it** in `tasks/README.md`.

---

## PR Checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Task files validate: `python run.py --dry-run`
- [ ] New tasks have unique `id` values
- [ ] New backends have `is_available()` + `discover_models()` implemented
- [ ] New scoring types have tests in `tests/test_scorer.py`
- [ ] No hardcoded paths or platform-specific code
- [ ] README updated if new flags or categories added

---

## Code Style

- Python 3.11+, type hints on all public functions
- No third-party dependencies beyond `openai`, `pyyaml`, `rich`, `requests`
- Keep `benchmark/` modules import-clean (no circular deps)
- `run.py` is the only entry point — keep it thin, logic lives in `benchmark/`

## Questions?

Open a [GitHub Discussion](https://github.com/ericrichardson/llm-benchmark/discussions) or file an [Issue](https://github.com/ericrichardson/llm-benchmark/issues).
