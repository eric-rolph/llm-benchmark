"""
benchmark/backends — pluggable inference backend abstraction.

Supported backends:
  lm_studio   — LM Studio local server (OpenAI-compatible, /v1)
  ollama      — Ollama (native /api/tags discovery, /v1 chat)
  llamacpp    — llama.cpp server (OpenAI-compatible, /v1)

Usage:
  from benchmark.backends import create_backend, discover_all_models
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


def discover_all_models(config: dict) -> list[ModelInfo]:
    """
    Probe all enabled backends and return a flat list of discovered ModelInfo.
    Models from manually-specified config['models'] are appended last,
    tagged to their backend if identifiable, else tagged 'manual'.
    """
    found: list[ModelInfo] = []
    found_ids: set[str] = set()

    backends_cfg: dict = config.get("backends", {})
    for backend_type, backend_cfg in backends_cfg.items():
        if not backend_cfg.get("enabled", False):
            continue
        if not backend_cfg.get("auto_discover", True):
            continue
        try:
            backend = create_backend(backend_type, backend_cfg)
            if not backend.is_available():
                continue
            for m in backend.discover_models():
                key = f"{backend_type}:{m.id}"
                if key not in found_ids:
                    found_ids.add(key)
                    found.append(m)
        except Exception:
            pass

    # Manual models from top-level 'models' list
    for mid in config.get("models", []):
        if mid not in {m.id for m in found}:
            found.append(ModelInfo(id=mid, name=mid, backend_name="manual",
                                   size_bytes=None, details={}))

    return found


__all__ = [
    "Backend", "ModelInfo",
    "LMStudioBackend", "OllamaBackend", "LlamaCppBackend",
    "VLLMBackend", "SGLangBackend", "TensorRTBackend", "TGIBackend",
    "KTransformersBackend", "GenericOpenAIBackend",
    "create_backend", "discover_all_models",
]
