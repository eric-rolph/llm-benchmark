"""
benchmark/backends/ollama.py — Ollama backend.

Discovery: GET /api/tags   → {"models": [{"name": "llama3.2:latest", ...}]}
Running:   GET /api/ps     → currently loaded models
Chat:      POST /v1/chat/completions  (OpenAI-compat; Ollama ≥ 0.1.24)
Thinking:  POST extra param "think": true (Ollama ≥ 0.7); <think> tags in content
"""
from __future__ import annotations

import requests

from benchmark.backends.base import Backend, ModelInfo


class OllamaBackend(Backend):

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self._api_root()}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def discover_models(self) -> list[ModelInfo]:
        try:
            r = requests.get(f"{self._api_root()}/api/tags", timeout=5)
            r.raise_for_status()
            raw_models = r.json().get("models", [])
            models = []
            for item in raw_models:
                mid = item.get("name", "")
                if not mid:
                    continue
                details = item.get("details", {})
                models.append(ModelInfo(
                    id=mid,
                    name=mid,
                    backend_name=self.name or "Ollama",
                    size_bytes=item.get("size"),
                    details={
                        "family": details.get("family", ""),
                        "parameter_size": details.get("parameter_size", ""),
                        "quantization_level": details.get("quantization_level", ""),
                        "format": details.get("format", ""),
                    },
                ))
            return models
        except Exception:
            return []

    def get_extra_chat_params(self, task: dict) -> dict:
        """
        Inject Ollama-specific parameters.  When a task sets "thinking: true"
        (or when enabled globally in config), pass think=True so Ollama's
        reasoning engine is activated.
        """
        params = {}
        if task.get("thinking") or self.config.get("thinking", False):
            params["think"] = True
        return params
