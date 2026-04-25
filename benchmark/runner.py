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
import requests
import subprocess
import threading
from rich.console import Console

from benchmark.backends.base import Backend
from benchmark.utils import strip_thinking, _avg


class TelemetryTracker:
    """Background thread that polls nvidia-smi for peak VRAM and GPU utilization."""
    def __init__(self):
        self.running = False
        self.thread = None
        self.peak_vram_mb = 0
        self.utils = []
        self._smi_available = True

    def _poll(self):
        while self.running and self._smi_available:
            try:
                # --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
                # Example output: "12543, 85\n12000, 70" (for multiple GPUs)
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.0
                )
                if proc.returncode == 0:
                    lines = proc.stdout.strip().split("\n")
                    total_vram = 0
                    total_util = 0
                    for line in lines:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                total_vram += int(parts[0].strip())
                                total_util += int(parts[1].strip())
                            except ValueError:
                                pass
                    self.peak_vram_mb = max(self.peak_vram_mb, total_vram)
                    if lines:
                        self.utils.append(total_util / len(lines)) # avg util across GPUs
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._smi_available = False
            time.sleep(0.5)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        result = {}
        if self.peak_vram_mb > 0:
            result["peak_vram_mb"] = self.peak_vram_mb
        if self.utils:
            result["avg_gpu_util"] = sum(self.utils) / len(self.utils)
        return result


class ModelRunner:
    """
    Runs benchmark tasks against a single model via the provided Backend.
    """

    def __init__(self, backend: Backend, model_id: str, bench_config: dict):
        self.backend = backend
        self.model_id = model_id
        self.bench = bench_config
        self._client = backend.get_openai_client()
        self.hf_generation_config = self._fetch_hf_config(model_id)

    def _fetch_hf_config(self, model_id: str) -> dict:
        """Attempt to fetch optimal generation parameters directly from Hugging Face."""
        if "/" not in model_id:
            return {}
        
        config = {}
        headers = {"User-Agent": "LLMBenchmarkSuite/1.0"}
        urls_to_check = [
            f"https://huggingface.co/{model_id}/raw/main/generation_config.json",
            f"https://huggingface.co/{model_id}/raw/main/config.json"
        ]
        
        for url in urls_to_check:
            for attempt in range(2):
                try:
                    r = requests.get(url, headers=headers, timeout=5)
                    if r.status_code == 200:
                        data = r.json()
                        for key in ["temperature", "top_p", "repetition_penalty", "max_new_tokens", "top_k"]:
                            if key in data and key not in config:
                                config[key] = data[key]
                        break  # Success, move to next URL (or exit if we have everything, but we'll check both just in case)
                except Exception:
                    if attempt == 0:
                        time.sleep(1) # wait before retry
            
        if config:
            Console().print(f"  [dim]↳ Auto-loaded generation params from HF for {model_id} (e.g. temp={config.get('temperature', 'N/A')}, rep_penalty={config.get('repetition_penalty', 'N/A')})[/dim]")
            
        return config

    # ── model loading ────────────────────────────────────────────────────────

    def ensure_model_loaded(self) -> None:
        self.backend.ensure_model_loaded(self.model_id)

    # ── task execution ───────────────────────────────────────────────────────

    def _run_once(self, task: dict) -> dict:
        messages: list[dict] = []
        if task.get("system"):
            messages.append({"role": "system", "content": task["system"]})
        # Prepend few-shot examples as conversation history before the actual prompt
        for example in task.get("few_shot", []):
            messages.append({"role": "user",      "content": str(example.get("user", ""))})
            messages.append({"role": "assistant",  "content": str(example.get("assistant", ""))})
        messages.append({"role": "user", "content": task["prompt"]})

        hf_cfg = getattr(self, "hf_generation_config", {})
        
        # Override order: Task > HuggingFace > Global Benchmark Config
        temperature = task.get("temperature", hf_cfg.get("temperature", self.bench.get("temperature", 0.0)))
        max_tokens  = task.get("max_tokens",  hf_cfg.get("max_new_tokens", self.bench.get("max_tokens", 4096)))
        req_timeout = self.bench.get("timeout", 180)

        # Pull other useful HF params if they exist
        top_p       = task.get("top_p", hf_cfg.get("top_p"))
        top_k       = task.get("top_k", hf_cfg.get("top_k"))
        rep_penalty = task.get("repetition_penalty", hf_cfg.get("repetition_penalty"))

        extra_params = self.backend.get_extra_chat_params(task)
        
        # extra_body passes non-standard kwargs (like top_k, repetition_penalty) to the server
        extra_body = extra_params.pop("extra_body", {})
        if top_k is not None:
            extra_body["top_k"] = top_k
        if rep_penalty is not None:
            extra_body["repetition_penalty"] = rep_penalty

        t_start = time.perf_counter()
        t_first_reasoning: float | None = None
        t_first_content:   float | None = None
        response_text  = ""
        reasoning_text = ""
        completion_tokens = 0
        reasoning_tokens  = 0
        
        telemetry = TelemetryTracker()
        telemetry.start()

        try:
            kwargs = {
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "timeout": req_timeout,
                **extra_params,
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            if extra_body:
                kwargs["extra_body"] = extra_body

            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta

                    # Reasoning/thinking field (LM Studio: DeepSeek, Qwen3, Kimi)
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        if t_first_reasoning is None:
                            t_first_reasoning = time.perf_counter()
                        reasoning_text += rc

                    # Ollama thinking field (returned when think=True)
                    ot = getattr(delta, "thinking", None)
                    if ot:
                        if t_first_reasoning is None:
                            t_first_reasoning = time.perf_counter()
                        reasoning_text += ot

                    # Regular content — the actual answer
                    if delta.content:
                        if t_first_content is None:
                            t_first_content = time.perf_counter()
                        response_text += delta.content

                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                    completion_tokens = usage.completion_tokens or 0
                    # Read reasoning_tokens from usage details when available
                    # (supported by OpenAI, OpenRouter, and LM Studio ≥ 0.3.6)
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        rt = getattr(details, "reasoning_tokens", None)
                        if rt is not None:
                            reasoning_tokens = int(rt)

        except Exception as e:
            telemetry.stop()
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

        telemetry_data = telemetry.stop()
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        # TTFT = earliest first-token timestamp (reasoning or content, whichever came first)
        candidates = [t for t in (t_first_reasoning, t_first_content) if t is not None]
        t_first = min(candidates) if candidates else None
        ttft_ms = (t_first - t_start) * 1000 if t_first else None

        # TPS: only report when we have a reliable completion_tokens count from the API;
        # word-count estimates are not comparable across models and are suppressed.
        tps: float | None = (
            round(completion_tokens / (total_ms / 1000), 1)
            if completion_tokens > 0 and total_ms > 0
            else None
        )

        # Strip any <think>…</think> blocks that may have leaked into delta.content
        clean = strip_thinking(response_text)

        return {
            "task_id": task["id"],
            "response": clean,
            "reasoning_preview": reasoning_text[:200] if reasoning_text else None,
            "error": None,
            "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
            "total_ms": round(total_ms, 1),
            "tps": tps,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "backend": self.backend.name,
            **telemetry_data,
        }

    def run_task(self, task: dict) -> dict:
        runs = self.bench.get("runs_per_task", 1)
        if runs <= 1:
            return self._run_once(task)

        results = [self._run_once(task) for _ in range(runs)]
        errors = [r for r in results if r.get("error")]
        # Use the last successful run as the base dict; fall back to last run if all failed
        last = next((r for r in reversed(results) if not r.get("error")), results[-1])
        avg_ttft  = _avg([r["ttft_ms"]  for r in results])
        avg_total = _avg([r["total_ms"] for r in results])
        avg_tps   = _avg([r["tps"]      for r in results])
        return {
            **last,
            "error":    errors[-1]["error"] if len(errors) == len(results) else None,
            "ttft_ms":  round(avg_ttft,  1) if avg_ttft  is not None else None,
            "total_ms": round(avg_total, 1) if avg_total is not None else None,
            "tps":      round(avg_tps,   1) if avg_tps   is not None else None,
            # All individual run results — used by run.py to compute score variance.
            # Prefixed with _ to signal it is internal/transient (not written to JSONL).
            "_all_runs": results,
        }

    def run_task_k(self, task: dict, k: int) -> list[dict]:
        """Run the task k independent times (used for pass@k scoring)."""
        return [self._run_once(task) for _ in range(k)]
