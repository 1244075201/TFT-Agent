"""Central environment-backed application configuration."""

from pathlib import Path
from typing import Literal

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
    vision_model: str = Field(default="deepseek-v4-flash-vision-exp", min_length=1)
    vision_detail: Literal["low", "high", "original", "auto"] = "high"
    thinking_mode: Literal["enabled", "disabled"] = "disabled"
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    decision_timeout_seconds: float = Field(default=12, ge=3, le=60)
    database_path: Path = Path("data/tft.sqlite3")
    current_game_patch: str = "18.1d"
    current_set_name: str = "Enchanted Wilds"
    static_data_version: str = "16.17"
    communitydragon_base_url: AnyHttpUrl = AnyHttpUrl(
        "https://raw.communitydragon.org/16.17/plugins/rcp-be-lol-game-data/global/zh_cn/v1/"
    )
    communitydragon_tft_url: AnyHttpUrl = AnyHttpUrl(
        "https://raw.communitydragon.org/16.17/cdragon/tft/zh_cn.json"
    )
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"
    retrieval_threshold: float = Field(default=0.78, ge=-1, le=1)
    datatft_url: AnyHttpUrl = AnyHttpUrl("https://www.datatft.com/comps/tier/")
    datatft_server: str = "cn"
    datatft_min_sync_hours: int = Field(default=6, ge=1, le=168)
    datatft_stale_hours: int = Field(default=48, ge=1, le=720)
    datatft_request_interval_seconds: float = Field(default=0.75, ge=0.25, le=10)
    experimental_live_enabled: bool = False
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
