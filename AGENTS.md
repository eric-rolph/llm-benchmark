# Agent Notes

This repository is a local-first LLM benchmark suite for comparing local and
hosted models on deterministic tasks, including coding, repo-patch, and
observed agent-loop workflows.

## Current Project Shape

- Main entry point: `python .\run.py` or installed `llm-bench`.
- Task files live in `tasks/`; fixture repos live under `tasks/fixtures/`.
- Current suite size is 118 tasks across 10 categories with 21 scoring modes.
- High-value coding surfaces are `coding`, `repo_patch`, `agent_loop`, and
  `agentic`.
- `agent_loop` tasks execute an observed tool loop over a copied fixture repo:
  list/read/write/run visible tests, then inject hidden tests after `final`.

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
reuse known-bad rows.

Recent signal from hard `agent_loop` probes:

- Claude Opus 4.7 and Qwen 3.7 Max passed both hard tasks.
- GLM 5.2 passed TTL cache and failed CSV hidden tests.
- Kimi K2.6 now executes native OpenRouter tool calls; latest retry passed TTL
  cache and failed CSV after emitting prose instead of an action.
- MiniMax M3 still emits prose instead of valid actions on the hard tasks.

## Harness Notes

- `benchmark/agent_loop.py` accepts JSON actions, function-call syntax,
  summary-only final JSON, OpenRouter/Kimi native `tool_calls`, Kimi tool-call
  section markup, and reasoning-only fields.
- Non-stream agent-loop runs should count
  `usage.completion_tokens_details.reasoning_tokens` when present.
- Hosted runs should preserve `prompt_tokens`, `completion_tokens`,
  `reasoning_tokens`, `total_tokens`, and provider-reported `api_cost` when
  available. OpenRouter exposes `usage.cost` on chat responses.
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
