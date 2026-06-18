import pytest
from benchmark.backends import create_backend
from benchmark.backends.generic_openai import GenericOpenAIBackend
from benchmark.backends.lm_studio import _EMBED_RE

def test_backend_registry_instantiation():
    """Ensure the backend registry can instantiate all backends without breaking."""
    mock_config = {"base_url": "http://localhost:8080"}
    
    # Test a legacy backend
    ollama = create_backend("ollama", mock_config)
    assert type(ollama).__name__ == "OllamaBackend"
    
    # Test new backends
    vllm = create_backend("vllm", mock_config)
    assert type(vllm).__name__ == "VLLMBackend"
    
    tgi = create_backend("tgi", mock_config)
    assert type(tgi).__name__ == "TGIBackend"
    
    sglang = create_backend("sglang", mock_config)
    assert type(sglang).__name__ == "SGLangBackend"

def test_backend_invalid_name():
    """Ensure invalid backend requests fail gracefully."""
    mock_config = {"base_url": "http://localhost:8080"}
    with pytest.raises(ValueError, match="Unknown backend type"):
        create_backend("nonexistent_engine", mock_config)


def test_lm_studio_embedding_fallback_does_not_reject_fable_coder_model():
    """The e5 embedding shortcut must not match model names like fable5."""
    assert _EMBED_RE.search("gemma-4-12b-coder-fable5-composer2.5-v1") is None
    assert _EMBED_RE.search("text-embedding-nomic-embed-text-v1.5") is not None


def test_openai_compatible_backend_uses_auth_header_for_availability(monkeypatch):
    seen = []

    class Response:
        status_code = 200

    def fake_get(url, **kwargs):
        seen.append((url, kwargs))
        return Response()

    monkeypatch.setenv("LLM_BENCH_GENERIC_OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("benchmark.backends.base.requests.get", fake_get)

    backend = GenericOpenAIBackend({
        "name": "Generic OpenAI",
        "base_url": "https://api.openai.com/v1",
    })

    assert backend.is_available() is True
    assert seen[0][0] == "https://api.openai.com/health"
    assert seen[0][1]["headers"] == {"Authorization": "Bearer test-key"}


def test_openai_compatible_backend_uses_auth_header_for_model_discovery(monkeypatch):
    seen = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "gpt-5.5"}]}

    def fake_get(url, **kwargs):
        seen.append((url, kwargs))
        return Response()

    monkeypatch.setenv("LLM_BENCH_GENERIC_OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("benchmark.backends.base.requests.get", fake_get)

    backend = GenericOpenAIBackend({
        "name": "Generic OpenAI",
        "base_url": "https://api.openai.com/v1",
    })

    models = backend.discover_models()

    assert [m.id for m in models] == ["gpt-5.5"]
    assert seen[0][0] == "https://api.openai.com/v1/models"
    assert seen[0][1]["headers"] == {"Authorization": "Bearer test-key"}


def test_openai_api_base_url_can_use_standard_openai_api_key(monkeypatch):
    monkeypatch.delenv("LLM_BENCH_GENERIC_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    backend = GenericOpenAIBackend({
        "name": "Generic OpenAI",
        "base_url": "https://api.openai.com/v1",
    })

    assert backend._auth_headers() == {"Authorization": "Bearer openai-key"}


def test_openrouter_api_base_url_can_use_standard_openrouter_api_key(monkeypatch):
    monkeypatch.delenv("LLM_BENCH_GENERIC_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    backend = GenericOpenAIBackend({
        "name": "Generic OpenAI",
        "base_url": "https://openrouter.ai/api/v1",
    })

    assert backend._auth_headers() == {"Authorization": "Bearer openrouter-key"}


def test_generic_openai_backend_passes_chat_extra_body_from_config_and_task():
    backend = GenericOpenAIBackend({
        "name": "Generic OpenAI",
        "base_url": "https://openrouter.ai/api/v1",
        "extra_body": {
            "reasoning": {"max_tokens": 512},
            "provider": {"sort": "throughput"},
        },
    })

    assert backend.get_extra_chat_params({}) == {
        "extra_body": {
            "reasoning": {"max_tokens": 512},
            "provider": {"sort": "throughput"},
        },
    }
    assert backend.get_extra_chat_params({
        "extra_body": {
            "reasoning": {"max_tokens": 1024},
        },
    }) == {
        "extra_body": {
            "reasoning": {"max_tokens": 1024},
            "provider": {"sort": "throughput"},
        },
    }


def test_generic_openai_backend_can_opt_into_responses_api():
    backend = GenericOpenAIBackend({
        "name": "Generic OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api": "responses",
        "reasoning_effort": "high",
        "text_verbosity": "low",
        "max_output_tokens": 25000,
    })

    task = {
        "id": "t1",
        "prompt": "solve",
        "reasoning_effort": "medium",
        "max_output_tokens": 12000,
    }

    assert backend.use_responses_api("gpt-5.5", task) is True
    assert backend.get_responses_params(task, {"max_tokens": 4096}) == {
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 12000,
    }
