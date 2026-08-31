"""Central environment-backed application configuration."""

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from tft_agent.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Configuration loaded from environment variables and an optional local .env."""

    model_config = SettingsConfigDict(
        env_prefix="TFT_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    max_steps: int = Field(default=8, ge=1, le=100)
    model: str = Field(default="deepseek-v4-flash", min_length=1)
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="DEEPSEEK_API_KEY",
    )
    deepseek_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://api.deepseek.com"),
        validation_alias="DEEPSEEK_BASE_URL",
    )

    def require_deepseek_api_key(self) -> SecretStr:
        """Return the configured key or fail before any provider request is attempted."""

        if self.deepseek_api_key is None or not self.deepseek_api_key.get_secret_value().strip():
            raise ConfigurationError("DEEPSEEK_API_KEY is not configured")
        return self.deepseek_api_key
