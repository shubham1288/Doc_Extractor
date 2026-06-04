"""Tests for structured logging configuration."""

import json
import logging
import os
from io import StringIO
from unittest.mock import patch

import structlog

from app.core.logging import get_logger, setup_logging


def _capture_log_output(log_action, log_format="json"):
    """Helper to capture structured log output reliably."""
    with patch.dict(os.environ, {"LOG_FORMAT": log_format}):
        # Reset structlog caching so each test gets fresh config
        structlog.reset_defaults()
        setup_logging()

    # Redirect root handler to a string buffer
    root = logging.getLogger()
    buffer = StringIO()
    root.handlers.clear()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(root.handlers[0].formatter if root.handlers else None)

    # Re-setup with our buffer
    with patch.dict(os.environ, {"LOG_FORMAT": log_format}):
        structlog.reset_defaults()
        setup_logging()

    # Replace the stdout handler with our buffer handler
    root = logging.getLogger()
    formatter = root.handlers[0].formatter
    root.handlers.clear()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    log_action()

    output = buffer.getvalue().strip()
    buffer.close()
    return output


class TestSetupLogging:
    """Tests for setup_logging() initialization."""

    def setup_method(self):
        """Reset structlog before each test."""
        structlog.reset_defaults()

    def test_setup_logging_configures_root_logger(self):
        """Root logger should have a handler after setup."""
        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert root.level == logging.INFO

    def test_setup_logging_respects_log_level_env(self):
        """LOG_LEVEL env var should control the root logger level."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            setup_logging()
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_logging_defaults_to_info(self):
        """Default log level should be INFO when LOG_LEVEL is not set."""
        env = os.environ.copy()
        env.pop("LOG_LEVEL", None)
        with patch.dict(os.environ, env, clear=True):
            setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_setup_logging_json_format_produces_json(self):
        """JSON format should produce parseable JSON log lines."""
        output = _capture_log_output(
            lambda: get_logger("test_json").info("test_event", key="value"),
            log_format="json",
        )
        log_entry = json.loads(output)
        assert log_entry["event"] == "test_event"
        assert log_entry["key"] == "value"
        assert log_entry["level"] == "info"
        assert "timestamp" in log_entry

    def test_setup_logging_console_format(self):
        """Console format should produce human-readable output."""
        output = _capture_log_output(
            lambda: get_logger("test_console").info("console_event"),
            log_format="console",
        )
        assert "console_event" in output

    def test_setup_logging_clears_existing_handlers(self):
        """setup_logging should clear pre-existing root handlers."""
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        assert len(root.handlers) >= 2

        setup_logging()
        assert len(root.handlers) == 1


class TestGetLogger:
    """Tests for get_logger() helper."""

    def setup_method(self):
        """Reset structlog before each test."""
        structlog.reset_defaults()

    def test_get_logger_returns_bound_logger(self):
        """get_logger should return a structlog BoundLogger."""
        setup_logging()
        logger = get_logger("my_module")
        assert logger is not None

    def test_get_logger_with_bindings(self):
        """get_logger with initial bindings should include them in output."""
        output = _capture_log_output(
            lambda: get_logger("bound_test", service="kyc", version="1.0").info("bound_event"),
            log_format="json",
        )
        log_entry = json.loads(output)
        assert log_entry["service"] == "kyc"
        assert log_entry["version"] == "1.0"

    def test_get_logger_without_name(self):
        """get_logger without a name should still return a logger."""
        setup_logging()
        logger = get_logger()
        assert logger is not None

    def test_logger_includes_timestamp_iso(self):
        """Log entries should include ISO-formatted timestamp."""
        output = _capture_log_output(
            lambda: get_logger("ts_test").info("timestamp_check"),
            log_format="json",
        )
        log_entry = json.loads(output)
        # ISO timestamp contains 'T' separator
        assert "T" in log_entry["timestamp"]

    def test_logger_includes_log_level(self):
        """Log entries should include the log level."""
        output = _capture_log_output(
            lambda: get_logger("level_test").warning("warn_event"),
            log_format="json",
        )
        log_entry = json.loads(output)
        assert log_entry["level"] == "warning"

    def test_stdlib_logging_integration(self):
        """Standard library loggers should produce structured output."""
        def log_action():
            stdlib_logger = logging.getLogger("third_party_lib")
            stdlib_logger.setLevel(logging.INFO)
            stdlib_logger.info("stdlib message")

        output = _capture_log_output(log_action, log_format="json")
        log_entry = json.loads(output)
        assert log_entry["event"] == "stdlib message"
        assert "timestamp" in log_entry
