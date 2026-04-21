"""
benchmark/runner.py — hits an inference backend and measures timing.

Thinking model support (Qwen3, Kimi K2, DeepSeek-R1, Nemotron, etc.):
  LM Studio streams internal reasoning in delta.reasoning_content.
  Ollama (with think=True) embeds thinking in delta.content as <think>…</think>.
  llama.cpp uses <think>…</think> tags directly.
  We capture reasoning separately and report clean answer text to the scorer.
"""
from __future__ import annotations

import time

from benchmark.backends.base import Backend
from benchmark.utils import strip_thinking


def _avg(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


class ModelRunner:
    """
    Runs benchmark tasks against a single model via the provided Backend.
    """

    def __init__(self, backend: Backend, model_id: str, bench_config: dict):
        self.backend = backend
        self.model_id = model_id
        self.bench = bench_config
        self._client = backend.get_openai_client()

    # ── model loading ────────────────────────────────────────────────────────

    def ensure_model_loaded(self) -> None:
        self.backend.ensure_model_loaded(self.model_id)

    # ── task execution ───────────────────────────────────────────────────────

    def _run_once(self, task: dict) -> dict:
        messages: list[dict] = []
        if task.get("system"):
            messages.append({"role": "system", "content": task["system"]})
        messages.append({"role": "user", "content": task["prompt"]})

        temperature = task.get("temperature", self.bench.get("temperature", 0.0))
        max_tokens  = task.get("max_tokens",  self.bench.get("max_tokens", 4096))
        req_timeout = self.bench.get("timeout", 180)

        extra_params = self.backend.get_extra_chat_params(task)

        t_start = time.perf_counter()
        t_first_reasoning: float | None = None
        t_first_content:   float | None = None
        response_text  = ""
        reasoning_text = ""
        completion_tokens = 0
        reasoning_tokens  = 0

        try:
            stream = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                timeout=req_timeout,
                **extra_params,
            )
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta

                    # Reasoning/thinking field (LM Studio: DeepSeek, Qwen3, Kimi)
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        if t_first_reasoning is None:
                            t_first_reasoning = time.perf_counter()
                        reasoning_text += rc
                        reasoning_tokens += 1

                    # Ollama thinking field (returned when think=True)
                    ot = getattr(delta, "thinking", None)
                    if ot:
                        if t_first_reasoning is None:
                            t_first_reasoning = time.perf_counter()
                        reasoning_text += ot
                        reasoning_tokens += 1

                    # Regular content — the actual answer
                    if delta.content:
                        if t_first_content is None:
                            t_first_content = time.perf_counter()
                        response_text += delta.content

                if getattr(chunk, "usage", None):
                    completion_tokens = chunk.usage.completion_tokens or 0

        except Exception as e:
            return {
                "task_id": task["id"],
                "response": "",
                "error": str(e),
                "ttft_ms": None,
                "total_ms": None,
                "tps": None,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "backend": self.backend.name,
            }

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        # TTFT = time to first *content* token; fall back to first reasoning token
        t_first = t_first_content or t_first_reasoning
        ttft_ms = (t_first - t_start) * 1000 if t_first else None

        # TPS measured over the content-generation window only
        t_content_start = t_first_content or t_first_reasoning or t_start
        gen_s = t_end - t_content_start
        answer_tokens = max(1, len(response_text.split())) if response_text else 1
        tok = completion_tokens if completion_tokens > 0 else answer_tokens
        tps = tok / gen_s if gen_s > 0.01 else 0.0

        # Strip any <think>…</think> blocks that may have leaked into delta.content
        clean = strip_thinking(response_text)

        return {
            "task_id": task["id"],
            "response": clean,
            "reasoning_preview": reasoning_text[:200] if reasoning_text else None,
            "error": None,
            "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
            "total_ms": round(total_ms, 1),
            "tps": round(tps, 1),
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "backend": self.backend.name,
        }

    def run_task(self, task: dict) -> dict:
        runs = self.bench.get("runs_per_task", 1)
        if runs <= 1:
            return self._run_once(task)

        results = [self._run_once(task) for _ in range(runs)]
        last = results[-1]
        return {
            **last,
            "ttft_ms":  round(_avg([r["ttft_ms"]  for r in results]) or 0, 1),
            "total_ms": round(_avg([r["total_ms"] for r in results]) or 0, 1),
            "tps":      round(_avg([r["tps"]       for r in results]) or 0, 1),
        }
