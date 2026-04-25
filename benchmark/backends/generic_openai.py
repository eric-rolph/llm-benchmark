"""
benchmark/backends/generic_openai.py — Generic OpenAI-compatible backend.
Can be used for custom Hugging Face Transformers API wrappers or any unknown server.
"""
from __future__ import annotations

from benchmark.backends.base import BaseOpenAIBackend

class GenericOpenAIBackend(BaseOpenAIBackend):
    pass
