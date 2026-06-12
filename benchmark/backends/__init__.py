"""
benchmark/backends — pluggable inference backend abstraction.

Supported backends:
  lm_studio   — LM Studio local server (OpenAI-compatible, /v1)
  ollama      — Ollama (native /api/tags discovery, /v1 chat)
  llamacpp    — llama.cpp server (OpenAI-compatible, /v1)

Usage:
  from benchmark.backends import create_backend
"""
from benchmark.backends.base import Backend, ModelInfo
from benchmark.backends.lm_studio import LMStudioBackend
from benchmark.backends.ollama import OllamaBackend
from benchmark.backends.llamacpp import LlamaCppBackend
from benchmark.backends.vllm import VLLMBackend
from benchmark.backends.sglang import SGLangBackend
from benchmark.backends.tensorrt import TensorRTBackend
from benchmark.backends.tgi import TGIBackend
from benchmark.backends.ktransformers import KTransformersBackend
from benchmark.backends.generic_openai import GenericOpenAIBackend

_REGISTRY: dict[str, type[Backend]] = {
    "lm_studio": LMStudioBackend,
    "ollama": OllamaBackend,
    "llamacpp": LlamaCppBackend,
    "vllm": VLLMBackend,
    "sglang": SGLangBackend,
    "tensorrt": TensorRTBackend,
    "tgi": TGIBackend,
    "ktransformers": KTransformersBackend,
    "generic_openai": GenericOpenAIBackend,
}


def create_backend(backend_type: str, config: dict) -> Backend:
    cls = _REGISTRY.get(backend_type)
    if cls is None:
        raise ValueError(f"Unknown backend type: {backend_type!r}. Valid types: {list(_REGISTRY)}")
    return cls(config)


__all__ = [
    "Backend", "ModelInfo",
    "LMStudioBackend", "OllamaBackend", "LlamaCppBackend",
    "VLLMBackend", "SGLangBackend", "TensorRTBackend", "TGIBackend",
    "KTransformersBackend", "GenericOpenAIBackend",
    "create_backend",
]
