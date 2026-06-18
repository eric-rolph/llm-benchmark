# Agent Loop Benchmark Implementation Plan

> **For agentic workers:** Historical implementation plan. The original plan
> used checkbox syntax for tracking; completed items are checked below.

**Goal:** Add a first objective `agent_loop` scoring surface that observes real iterative tool use against a local repository fixture.

**Architecture:** `ModelRunner` routes `scoring.type: agent_loop` to a new `benchmark.agent_loop` executor. The executor copies a fixture repo, lets the model request JSON tool actions, executes those actions, feeds observations back to the model, then injects hidden tests and scores the final workspace. `score_response` reads the direct score from the run result, while JSON/CSV persistence uses the existing execution trace field.

**Tech Stack:** Python 3.11, OpenAI-compatible Chat Completions, pytest fixture repos, existing repo-patch path/env/test-command helpers.

**Status:** Implemented. This plan is retained as historical design context;
the checkbox items below reflect completed development work. Optional live model
smoke testing is a manual validation step, not an outstanding implementation
task.

---

### Task 1: Red Tests For Observed Agent Loop

**Files:**
- Create: `tests/test_agent_loop.py`
- Modify: `tests/test_scorer.py`

- [x] **Step 1: Write failing executor tests**

Create tests with a fake chat client that returns JSON actions:

```python
{"tool": "read_file", "args": {"path": "calc/stats.py"}}
{"tool": "write_file", "args": {"path": "calc/stats.py", "content": "def mean(values):\n    return sum(values) / len(values)\n"}}
{"tool": "run_tests", "args": {}}
{"tool": "final", "args": {"summary": "fixed mean"}}
```

Assert the executor copies the fixture, records tool events, runs visible tests during the loop, injects hidden tests after final, and returns `agent_loop_score == 1.0`.

- [x] **Step 2: Add failure/security tests**

Add tests for unsafe paths, malformed JSON actions, max-step exhaustion, and disabled scoring via `score_response`.

- [x] **Step 3: Run tests to verify red**

Run: `python -m pytest tests/test_agent_loop.py tests/test_scorer.py::TestAgentLoop -q`

Expected: import errors or unknown scoring failures because `benchmark.agent_loop` and `agent_loop` scoring do not exist yet.

### Task 2: Minimal Agent Loop Executor

**Files:**
- Create: `benchmark/agent_loop.py`
- Modify: `benchmark/runner.py`
- Modify: `benchmark/scorer.py`
- Modify: `benchmark/session.py`

- [x] **Step 1: Implement `benchmark.agent_loop.run_agent_loop`**

Support tools: `list_files`, `read_file`, `write_file`, `run_tests`, `final`. Use repo-patch helpers for fixture resolution, safe path handling, sanitized test commands, and hidden-test execution.

- [x] **Step 2: Route runner calls**

In `ModelRunner._run_once`, detect `scoring.type == "agent_loop"` and call `run_agent_loop`.

- [x] **Step 3: Gate local execution**

In `session.run_model`, if an `agent_loop` task is selected without `--allow-code-exec`, return a disabled result instead of invoking the runner.

- [x] **Step 4: Score direct run results**

In `score_response`, special-case `agent_loop` by reading `agent_loop_score` and `agent_loop_detail` from `run_result`.

- [x] **Step 5: Run red tests to verify green**

Run: `python -m pytest tests/test_agent_loop.py tests/test_scorer.py::TestAgentLoop -q`

Expected: all selected tests pass.

### Task 3: Task Fixture And Docs

**Files:**
- Create: `tasks/fixtures/agent_loop/mean_tool_loop/...`
- Create: `tasks/agent_loop.yaml`
- Modify: `README.md`
- Modify: `tasks/README.md`
- Modify: `benchmark/evaluation.py`

- [x] **Step 1: Add one fixture task**

Create a tiny repo with a visible integer-mean test and a hidden fractional-mean test.

- [x] **Step 2: Add category metadata**

Add `agent_loop` to category weights and E3 expected tokens. Mark task `benchmark_tier: repo_agent` and `execution_surface: observed_agent_loop`.

- [x] **Step 3: Document accepted protocol**

Document JSON action format and available tools in `tasks/README.md` and README category tables.

- [x] **Step 4: Verify loader**

Run: `python run.py --dry-run --category agent_loop`

Expected: agent-loop task loads and validates.

### Task 4: Full Verification

**Files:**
- All touched files

- [x] **Step 1: Run targeted tests**

Run: `python -m pytest tests/test_agent_loop.py tests/test_scorer.py tests/test_runner_stream.py tests/test_result_schema.py -q`

- [x] **Step 2: Run full suite**

Run: `python -m pytest -q`

- [x] **Step 3: Run dry-run**

Run: `python run.py --dry-run --backend lm_studio`

- **Optional live smoke:** deferred manual validation with a selected local model.

Run: `python run.py --backend lm_studio --model "gemma-4-12b-coder-fable5-composer2.5-v1" --category agent_loop --allow-code-exec --output results\codex_agent_loop_smoke`

Expected: the task runs without harness crashes, whether the model succeeds or fails.
