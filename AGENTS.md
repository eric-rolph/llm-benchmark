# Agent Notes

This repository is a local-first LLM benchmark suite for comparing local and
hosted models on deterministic tasks, including coding, repo-patch, and
observed agent-loop workflows.

## Current Project Shape

- Main entry point: `python .\run.py` or installed `llm-bench`.
- Task files live in `tasks/`; fixture repos live under `tasks/fixtures/`.
- Current suite size is 119 tasks across 10 categories with 21 scoring modes.
- High-value coding surfaces are `coding`, `repo_patch`, `agent_loop`, and
  `agentic`.
- `agent_loop` tasks execute an observed tool loop over a copied fixture repo:
  list/read/write/run visible tests, then inject hidden tests after `final`.
- `repo_patch` and `agent_loop` reject model-authored edits to harness-control
  paths such as `tests/**`, `conftest.py`, pytest config files,
  `pyproject.toml`, and `sitecustomize.py`.

## Secrets

- Do not commit API keys.
- Local secrets belong in `.secrets/`, which is ignored by git.
- The OpenRouter key is expected at `.secrets/openrouter.env` as:
  `OPENROUTER_API_KEY=...`
- `configs/openrouter-frontier-agent-loop.yaml` loads that file via
  `secrets_file: "../.secrets/openrouter.env"`.
- Do not print secret values in logs or final responses. Verify presence with
  `set/unset` style checks only.

## OpenRouter Frontier Probe

Use this profile for hosted SOTA coding/agent-loop validation:

```powershell
python .\run.py --config .\configs\openrouter-frontier-agent-loop.yaml --backend generic_openai --category agent_loop --task agent_loop_004_csv_import_reconciliation agent_loop_005_ttl_cache_invalidation --allow-code-exec --output results\openrouter_frontier_agent_loop_hard_probe
```

The profile currently covers:

- `anthropic/claude-opus-4.7`
- `z-ai/glm-5.2`
- `google/gemini-3.5-flash`
- `moonshotai/kimi-k2.6`
- `qwen/qwen3.7-max`
- `minimax/minimax-m3`

Keep paid retries narrow. Prefer retrying only affected model/task pairs after
harness changes, and use a fresh output directory when `resume: true` could
reuse known-bad rows. The profile sets `benchmark.max_api_cost: 5.00`; override
with `--max-api-cost USD` only for intentional larger runs.
The profile also sets `benchmark.agent_loop_native_tools: true`, so OpenRouter
Chat Completions providers receive native function schemas for the five
`agent_loop` tools.

Recent signal from hard `agent_loop` probes:

- GPT-5.5 passed both hard tasks through the OpenAI Responses text-action path;
  future GPT-style reruns should set `agent_loop_native_tools: true` to use the
  native Responses API function-call path added after that probe.
- GLM 5.2 passed both hard tasks through OpenRouter Chat Completions.
- Kimi K2.6 passed TTL cache and failed CSV after emitting prose instead of an
  action.
- Claude Opus 4.7, Gemini 3.5 Flash, Qwen 3.7 Max, and MiniMax M3 all failed
  the two-task slice in the latest run, mostly through tool-protocol or
  no-final failures rather than proven inability to implement the fixes.
- Gemma 4 12B Coder passed the medium mean-repair smoke and failed both hard
  tasks; Gemma 4 12B QAT and base were 0/3 on the same local slice.

## Harness Notes

- `benchmark/agent_loop.py` accepts JSON actions, function-call syntax,
  summary-only final JSON, OpenRouter/Kimi native `tool_calls`, native OpenAI
  Responses API `function_call` items, Kimi tool-call section markup, and
  reasoning-only fields.
- Agent-loop rows keep binary hidden-test `score` stable and add
  `agent_loop_progress_score`, passed/total milestone counts, termination, and
  `execution_trace.progress` so protocol compliance and implementation progress
  can be analyzed separately from hidden-test pass/fail.
- `benchmark.agent_loop_native_tools: true` adds Chat Completions function
  schemas or Responses API function tools for `list_files`, `read_file`,
  `write_file`, `run_tests`, and `final`; the default remains the text-action
  parser for local compatibility.
- Non-stream agent-loop runs should count
  `usage.completion_tokens_details.reasoning_tokens` when present.
- Hosted runs should preserve `prompt_tokens`, `completion_tokens`,
  `reasoning_tokens`, `total_tokens`, and provider-reported `api_cost` when
  available. OpenRouter exposes `usage.cost` on chat responses.
- For `runs_per_task > 1`, usage and `api_cost` should be summed across all
  samples; `sample_count` and `score_std` are persisted for analysis.
- Resume cache keys include a secret-redacted run fingerprint built from model,
  backend/config, benchmark generation settings, code-exec mode, and judge
  model. Changing incompatible run settings should miss cached JSONL rows.
- Reports should show completed/intended task coverage. Composite scores should
  be suppressed for incomplete matrices so budget-stopped runs are not ranked
  as if they were complete.
- `--compare` only treats rows as shared when model, task id, and any recorded
  task version/hash metadata match; changed task definitions should be
  unmatched, not scored as regressions/improvements.
- `benchmark.max_api_cost` / `--max-api-cost` caps newly executed hosted work
  using accumulated provider-reported `api_cost`; cached resume rows do not
  consume the current run budget.
- Arena mode counts provider-reported judge `api_cost` toward the same budget
  and stores `judge_api_cost` in match history when available.
- For OpenRouter Chat Completions provider options, use backend or task-level
  `extra_body`, for example:

```yaml
extra_body:
  reasoning:
    max_tokens: 512
```

## Safety

- `--allow-code-exec` runs model-generated code locally. On Windows, memory
  limits are not enforced by `RLIMIT_AS`; keep fixtures trusted and scoped.
- Do not run broad paid frontier sweeps unless the user explicitly asks.
- Do not store real keys in `config.yaml`, README, task YAML, result notes, or
  committed docs.

## Verification

Before claiming completion after code changes, run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\run.py --dry-run --backend lm_studio
$env:OPENROUTER_API_KEY=$null; .\.venv\Scripts\python.exe .\run.py --config .\configs\openrouter-frontier-agent-loop.yaml --backend generic_openai --dry-run
git diff --check
```

For parser or backend changes, add targeted regression tests first, then run the
focused test file before the full suite.
