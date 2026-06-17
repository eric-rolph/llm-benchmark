"""
benchmark/backends/base.py — abstract backend interface.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse
import re

import requests
from openai import OpenAI


@dataclass
class ModelInfo:
    id: str
    name: str
    backend_name: str
    size_bytes: int | None = None
    details: dict = field(default_factory=dict)

    def label(self) -> str:
        size_str = ""
        if self.size_bytes:
            gb = self.size_bytes / 1e9
            size_str = f"  {gb:.1f} GB"
        return f"{self.name}{size_str}  [{self.backend_name}]"


class Backend(ABC):
    """
    Pluggable inference backend.  Each subclass knows how to:
      • check if the server is reachable (is_available)
      • list available models (discover_models)
      • return a configured OpenAI client for /v1/chat/completions
      • optionally ensure a model is loaded before first use
    """

    # True when the backend honors an explicit per-request thinking toggle
    # (used by --ab-thinking to run each task with thinking on vs off).
    supports_thinking_ab: bool = False

    def __init__(self, config: dict):
        self.config = config
        self.name: str = config.get("name", self.__class__.__name__)
        self.base_url: str = config.get("base_url", "")

    # ── required ─────────────────────────────────────────────────────────────

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend server is reachable."""

    @abstractmethod
    def discover_models(self) -> list[ModelInfo]:
        """Return all models currently available on this backend."""

    # ── optional overrides ───────────────────────────────────────────────────

    def get_openai_client(self) -> OpenAI:
        """Return an OpenAI client pointed at this backend's /v1 endpoint."""
        return OpenAI(base_url=self._v1_url(), api_key=self._api_key())

    def _api_key(self) -> str:
        """Return the configured API key, preferring the backend env var."""
        # Sanitize backend name for bash-compatible environment variables
        # (e.g. "llama.cpp" -> "LLAMA_CPP", "LM Studio" -> "LM_STUDIO")
        safe_name = re.sub(r'[^A-Z0-9]', '_', self.name.upper())
        env_key = os.environ.get(f"LLM_BENCH_{safe_name}_API_KEY")
        if env_key:
            return env_key

        if "api.openai.com" in self.base_url:
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                return openai_key

        return self.config.get("api_key", "noop")

    def _auth_headers(self) -> dict[str, str]:
        api_key = self._api_key()
        if not api_key:
            return {}
        return {"Authorization": f"Bearer {api_key}"}

    def get_extra_chat_params(self, task: dict) -> dict:
        """
        Extra keyword args injected into client.chat.completions.create().
        Override in backends that need non-standard params (e.g. Ollama think=True).
        """
        return {}

    def use_responses_api(self, model_id: str, task: dict) -> bool:
        """Return True when this backend should use client.responses.create()."""
        api_mode = str(self.config.get("api") or "").strip().lower()
        return bool(self.config.get("use_responses_api")) or api_mode == "responses"

    def get_responses_params(self, task: dict, bench_config: dict) -> dict:
        """Build optional Responses API request params from backend/task config."""
        params: dict = {}
        effort = task.get("reasoning_effort", self.config.get("reasoning_effort"))
        summary = task.get("reasoning_summary", self.config.get("reasoning_summary"))
        if effort or summary:
            reasoning = {}
            if effort:
                reasoning["effort"] = effort
            if summary:
                reasoning["summary"] = summary
            params["reasoning"] = reasoning

        verbosity = task.get("text_verbosity", self.config.get("text_verbosity"))
        if verbosity:
            params["text"] = {"verbosity": verbosity}

        max_output_tokens = task.get(
            "max_output_tokens",
            self.config.get(
                "max_output_tokens",
                bench_config.get("max_output_tokens", bench_config.get("max_tokens")),
            ),
        )
        if max_output_tokens:
            params["max_output_tokens"] = int(max_output_tokens)
        return params

    def ensure_model_loaded(self, model_id: str) -> None:
        """
        Ensure the model is loaded before inference begins.
        Default: no-op (Ollama and llama.cpp auto-load).
        """

    # ── helpers ──────────────────────────────────────────────────────────────

    def _v1_url(self) -> str:
        """
        Return the canonical /v1 base URL for the OpenAI client.
        Uses urllib.parse to reliably check the path component rather than
        a fragile string.endswith() check.
        """
        parsed = urlparse(self.base_url.rstrip("/"))
        # If the path already ends with /v1 (exact, not /v1/something), keep it.
        # Otherwise append /v1.
        path = parsed.path.rstrip("/")
        if not path.endswith("/v1"):
            path = path + "/v1"
        return urlunparse(parsed._replace(path=path))

    def _api_root(self) -> str:
        """Return base URL with any trailing /v1 stripped."""
        url = self.base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        return url.rstrip("/")


class BaseOpenAIBackend(Backend):
    """
    Standard implementation for backends that expose a generic OpenAI-compatible
    API with /health and /v1/models endpoints (e.g. vLLM, TGI, SGLang).
    """

    def is_available(self) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                r = requests.get(
                    f"{self._api_root()}{path}",
                    headers=self._auth_headers(),
                    timeout=3,
                )
                if r.status_code == 200:
                    return True
            except Exception:
                pass
        return False

    def discover_models(self) -> list[ModelInfo]:
        try:
            r = requests.get(
                f"{self._v1_url()}/models",
                headers=self._auth_headers(),
                timeout=5,
            )
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
                    backend_name=self.name or self.__class__.__name__,
                    details=item,
                ))
            return models
        except Exception:
            return []
