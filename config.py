from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_title: str = "Multi-lang Summarizer API"
    api_version: str = "1.0.0"
    api_description: str = "Production-ready translation and summarization service"

    cohere_api_key: str | None = None
    cohere_model: str = "command-a-03-2025"
    cohere_temperature: float = 0.1
    cohere_max_tokens: int = 1800

    allow_origins: List[str] = Field(default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"])
    requests_per_minute: int = 60

    max_text_chars: int = 120_000
    max_file_size_mb: int = 20
    max_audio_size_mb: int = 50

    speech_backend: str = "faster_whisper"
    whisper_model_size: str = "small"
    whisper_model_size_persian: str = "medium"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    whisper_temperature: float = 0.0
    whisper_vad_filter: bool = True
    whisper_condition_on_previous_text: bool = False

    cache_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: str | List[str]):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
