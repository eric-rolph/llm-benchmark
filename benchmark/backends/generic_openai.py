"""
benchmark/backends/generic_openai.py — Generic OpenAI-compatible backend.
Can be used for custom Hugging Face Transformers API wrappers or any unknown server.
"""
from __future__ import annotations

import requests

from benchmark.backends.base import Backend, ModelInfo


class GenericOpenAIBackend(Backend):

    def is_available(self) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                r = requests.get(f"{self._api_root()}{path}", timeout=3)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
        return False

    def discover_models(self) -> list[ModelInfo]:
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
                    backend_name=self.name or "GenericOpenAI",
                    details=item,
                ))
            return models
        except Exception:
            return []
