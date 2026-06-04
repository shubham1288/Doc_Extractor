"""Tests for Pydantic v2 settings configuration."""

import os
from unittest.mock import patch

from app.core.config import Settings, get_settings


class TestSettingsDefaults:
    """Verify default values for all settings fields."""

    def test_app_name_default(self):
        settings = Settings()
        assert settings.app_name == "KYC Document Extraction API"

    def test_app_version_default(self):
        settings = Settings()
        assert settings.app_version == "1.0.0"

    def test_debug_default(self):
        settings = Settings()
        assert settings.debug is False

    def test_host_default(self):
        settings = Settings()
        assert settings.host == "0.0.0.0"

    def test_port_default(self):
        settings = Settings()
        assert settings.port == 8000

    def test_log_level_default(self):
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_log_format_default(self):
        settings = Settings()
        assert settings.log_format == "json"

    def test_max_file_size_mb_default(self):
        settings = Settings()
        assert settings.max_file_size_mb == 10

    def test_max_pdf_pages_default(self):
        settings = Settings()
        assert settings.max_pdf_pages == 10

    def test_ocr_languages_default(self):
        settings = Settings()
        assert settings.ocr_languages == [
            "en", "hi", "mr", "gu", "bn", "ta", "te", "kn", "ml", "pa"
        ]

    def test_ocr_use_gpu_default(self):
        settings = Settings()
        assert settings.ocr_use_gpu is False

    def test_cors_origins_default(self):
        settings = Settings()
        assert settings.cors_origins == ["*"]

    def test_low_confidence_threshold_default(self):
        settings = Settings()
        assert settings.low_confidence_threshold == 0.3

    def test_classification_threshold_default(self):
        settings = Settings()
        assert settings.classification_threshold == 0.5


class TestSettingsEnvOverride:
    """Verify environment variable overrides with KYC_ prefix."""

    def test_port_override(self):
        with patch.dict(os.environ, {"KYC_PORT": "9000"}):
            settings = Settings()
        assert settings.port == 9000

    def test_debug_override(self):
        with patch.dict(os.environ, {"KYC_DEBUG": "true"}):
            settings = Settings()
        assert settings.debug is True

    def test_host_override(self):
        with patch.dict(os.environ, {"KYC_HOST": "127.0.0.1"}):
            settings = Settings()
        assert settings.host == "127.0.0.1"

    def test_log_level_override(self):
        with patch.dict(os.environ, {"KYC_LOG_LEVEL": "DEBUG"}):
            settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_max_file_size_mb_override(self):
        with patch.dict(os.environ, {"KYC_MAX_FILE_SIZE_MB": "25"}):
            settings = Settings()
        assert settings.max_file_size_mb == 25

    def test_ocr_use_gpu_override(self):
        with patch.dict(os.environ, {"KYC_OCR_USE_GPU": "true"}):
            settings = Settings()
        assert settings.ocr_use_gpu is True

    def test_low_confidence_threshold_override(self):
        with patch.dict(os.environ, {"KYC_LOW_CONFIDENCE_THRESHOLD": "0.5"}):
            settings = Settings()
        assert settings.low_confidence_threshold == 0.5

    def test_classification_threshold_override(self):
        with patch.dict(os.environ, {"KYC_CLASSIFICATION_THRESHOLD": "0.7"}):
            settings = Settings()
        assert settings.classification_threshold == 0.7

    def test_cors_origins_override_json(self):
        with patch.dict(os.environ, {"KYC_CORS_ORIGINS": '["http://localhost:3000","http://example.com"]'}):
            settings = Settings()
        assert settings.cors_origins == ["http://localhost:3000", "http://example.com"]

    def test_ocr_languages_override_json(self):
        with patch.dict(os.environ, {"KYC_OCR_LANGUAGES": '["en","hi"]'}):
            settings = Settings()
        assert settings.ocr_languages == ["en", "hi"]

    def test_app_name_override(self):
        with patch.dict(os.environ, {"KYC_APP_NAME": "Custom API"}):
            settings = Settings()
        assert settings.app_name == "Custom API"


class TestGetSettings:
    """Verify the get_settings() singleton behavior."""

    def test_get_settings_returns_settings_instance(self):
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_returns_same_instance(self):
        get_settings.cache_clear()
        first = get_settings()
        second = get_settings()
        assert first is second

    def test_get_settings_caches_result(self):
        get_settings.cache_clear()
        get_settings()
        info = get_settings.cache_info()
        assert info.currsize == 1
