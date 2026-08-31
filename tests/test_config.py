from pathlib import Path

import pytest
from pydantic import SecretStr

from tft_agent.core.config import Settings
from tft_agent.core.exceptions import ConfigurationError


def test_deepseek_defaults_and_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-placeholder")
    settings = Settings(_env_file=None)

    assert settings.model == "deepseek-v4-flash"
    assert str(settings.deepseek_base_url) == "https://api.deepseek.com/"
    assert isinstance(settings.require_deepseek_api_key(), SecretStr)


def test_missing_deepseek_key_raises_clear_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = Settings(_env_file=tmp_path / "missing.env")

    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY is not configured"):
        settings.require_deepseek_api_key()
