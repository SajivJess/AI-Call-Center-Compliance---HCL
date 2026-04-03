from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Call Center Compliance API"
    app_env: str = "dev"
    api_key: str = Field(default="change-me", alias="API_KEY")

    rate_limit_per_minute: int = 30
    request_timeout_seconds: int = 90

    ffmpeg_binary: str = "ffmpeg"
    temp_dir: str = "./tmp"

    sarvam_api_key: str | None = None
    sarvam_base_url: str = "https://api.sarvam.ai"

    whisper_api_key: str | None = None
    whisper_model: str = "whisper-1"

    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"

    sqlite_path: str = "./data/call_analytics.db"

    allow_mock_stt: bool = False
    allow_mock_llm: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
