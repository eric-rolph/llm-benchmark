import pytest
from benchmark.backends import get_backend

def test_backend_registry_instantiation():
    """Ensure the backend registry can instantiate all backends without breaking."""
    mock_config = {"base_url": "http://localhost:8080"}
    
    # Test a legacy backend
    ollama = get_backend("ollama", mock_config)
    assert ollama.name == "ollama"
    
    # Test new backends
    vllm = get_backend("vllm", mock_config)
    assert vllm.name == "vllm"
    
    tgi = get_backend("tgi", mock_config)
    assert tgi.name == "tgi"
    
    sglang = get_backend("sglang", mock_config)
    assert sglang.name == "sglang"

def test_backend_invalid_name():
    """Ensure invalid backend requests fail gracefully."""
    mock_config = {"base_url": "http://localhost:8080"}
    with pytest.raises(ValueError, match="Unknown backend type"):
        get_backend("nonexistent_engine", mock_config)
