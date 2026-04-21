"""
benchmark/backends/base.py — abstract backend interface.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

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
        v1_url = self._v1_url()
        # Allow env var override so real API keys are never committed to config
        api_key = os.environ.get(
            f"LLM_BENCH_{self.name.upper().replace(' ', '_')}_API_KEY",
            self.config.get("api_key", "noop"),
        )
        return OpenAI(base_url=v1_url, api_key=api_key)

    def get_extra_chat_params(self, task: dict) -> dict:
        """
        Extra keyword args injected into client.chat.completions.create().
        Override in backends that need non-standard params (e.g. Ollama think=True).
        """
        return {}

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
