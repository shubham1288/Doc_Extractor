"""
Application configuration using Pydantic v2 settings.

All settings can be overridden via environment variables prefixed with KYC_.
For example, KYC_PORT=9000 overrides the default port of 8000.

Usage:
    from app.core.config import get_settings

    settings = get_settings()
    print(settings.port)
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables with KYC_ prefix."""

    model_config = SettingsConfigDict(env_prefix="KYC_")

    app_name: str = "KYC Document Extraction API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_format: str = "json"
    max_file_size_mb: int = 10
    max_pdf_pages: int = 10
    ocr_languages: list[str] = [
        "en", "hi", "mr", "gu", "bn", "ta", "te", "kn", "ml", "pa"
    ]
    ocr_use_gpu: bool = False
    cors_origins: list[str] = ["*"]
    low_confidence_threshold: float = 0.3
    classification_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Uses lru_cache so the settings object is created once and reused
    across the application lifetime.
    """
    return Settings()
