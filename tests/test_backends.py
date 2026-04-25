import pytest
from benchmark.backends import create_backend

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
