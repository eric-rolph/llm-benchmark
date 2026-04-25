"""
benchmark/backends/llamacpp.py — llama.cpp server backend.

The llama.cpp server exposes an OpenAI-compatible API.

Discovery: GET /v1/models  (typically returns a single model)
Health:    GET /health     → {"status": "ok"}
Chat:      POST /v1/chat/completions
Thinking:  <think>…</think> tags embedded in delta.content
"""
from __future__ import annotations

from benchmark.backends.base import BaseOpenAIBackend


class LlamaCppBackend(BaseOpenAIBackend):
    def __init__(self, config: dict):
        super().__init__(config)
        if "name" not in config:
            self.name = "llama.cpp"
