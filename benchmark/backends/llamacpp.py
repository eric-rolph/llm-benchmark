"""
benchmark/backends/llamacpp.py — llama.cpp server backend.

The llama.cpp server exposes an OpenAI-compatible API.

Discovery: GET /v1/models  (typically returns a single model)
Health:    GET /health     → {"status": "ok"}
Chat:      POST /v1/chat/completions
Thinking:  <think>…</think> tags embedded in delta.content
"""
from __future__ import annotations

import requests

from benchmark.backends.base import Backend, ModelInfo


class LlamaCppBackend(Backend):

    def is_available(self) -> bool:
        # Try /health first, then /v1/models
        for path in ("/health", "/v1/models"):
            try:
                r = requests.get(f"{self._api_root()}{path}", timeout=3)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
        return False

    def discover_models(self) -> list[ModelInfo]:
        """
        llama.cpp usually serves one model at a time.
        We read its /v1/models list and return whatever is loaded.
        """
        try:
            r = requests.get(f"{self._v1_url()}/models", timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [])
            models = []
            for item in data:
                mid = item.get("id", "")
                if not mid:
                    continue
                models.append(ModelInfo(
                    id=mid,
                    name=mid,
                    backend_name=self.name or "llama.cpp",
                    details=item,
                ))
            return models
        except Exception:
            return []
