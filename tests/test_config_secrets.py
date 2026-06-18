import os

from benchmark.session import load_config


def test_load_config_loads_relative_secrets_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir()
    (secrets_dir / "openrouter.env").write_text(
        "OPENROUTER_API_KEY=test-openrouter-key\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'secrets_file: ".secrets/openrouter.env"\n'
        "backends: {}\n",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config["secrets_file"] == ".secrets/openrouter.env"
    assert os.environ["OPENROUTER_API_KEY"] == "test-openrouter-key"


def test_load_config_secrets_file_does_not_override_existing_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "existing-key")
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir()
    (secrets_dir / "openrouter.env").write_text(
        "OPENROUTER_API_KEY=file-key\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'secrets_file: ".secrets/openrouter.env"\n',
        encoding="utf-8",
    )

    load_config(str(config_path))

    assert os.environ["OPENROUTER_API_KEY"] == "existing-key"
