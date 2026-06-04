"""
Unit tests for PII filter and request logging middleware.

Tests:
- PII filter redacts Aadhaar patterns
- PII filter redacts PAN patterns
- PII filter handles non-string values
- Request logging middleware generates request_id
- Request logging middleware logs timing
"""

import pytest
from unittest.mock import patch

from app.middleware.pii_filter import (
    AADHAAR_PATTERN,
    PAN_PATTERN,
    REDACTED_AADHAAR,
    REDACTED_PAN,
    _redact_value,
    redact_pii,
)


class TestPIIFilterRedactValue:
    """Tests for _redact_value helper function."""

    def test_redacts_aadhaar_with_spaces(self):
        """Aadhaar number with spaces is redacted."""
        result = _redact_value("Number is 1234 5678 9012")
        assert REDACTED_AADHAAR in result
        assert "1234 5678 9012" not in result

    def test_redacts_aadhaar_without_spaces(self):
        """Aadhaar number without spaces is redacted."""
        result = _redact_value("Number is 123456789012")
        assert REDACTED_AADHAAR in result
        assert "123456789012" not in result

    def test_redacts_pan_number(self):
        """PAN number is redacted."""
        result = _redact_value("PAN: ABCDE1234F")
        assert REDACTED_PAN in result
        assert "ABCDE1234F" not in result

    def test_non_string_returns_unchanged(self):
        """Non-string values pass through unchanged."""
        assert _redact_value(42) == 42
        assert _redact_value(3.14) == 3.14
        assert _redact_value(None) is None
        assert _redact_value(True) is True

    def test_empty_string_returns_empty(self):
        """Empty string returns empty."""
        assert _redact_value("") == ""

    def test_text_without_pii_unchanged(self):
        """Text without PII patterns is not modified."""
        text = "Hello world, processing document type aadhaar"
        assert _redact_value(text) == text

    def test_multiple_aadhaar_patterns(self):
        """Multiple Aadhaar numbers in one string are all redacted."""
        text = "First: 1234 5678 9012, Second: 9876 5432 1098"
        result = _redact_value(text)
        assert not AADHAAR_PATTERN.search(result)
        assert result.count(REDACTED_AADHAAR) == 2

    def test_multiple_pan_patterns(self):
        """Multiple PAN numbers in one string are all redacted."""
        text = "PAN1: ABCDE1234F, PAN2: ZYXWV9876A"
        result = _redact_value(text)
        assert not PAN_PATTERN.search(result)
        assert result.count(REDACTED_PAN) == 2


class TestPIIFilterProcessor:
    """Tests for redact_pii structlog processor."""

    def test_redacts_event_field(self):
        """PII in event field is redacted."""
        event_dict = {"event": "Found Aadhaar 1234 5678 9012"}
        result = redact_pii(None, "info", event_dict)
        assert REDACTED_AADHAAR in result["event"]
        assert "1234 5678 9012" not in result["event"]

    def test_redacts_arbitrary_fields(self):
        """PII in any string field is redacted."""
        event_dict = {
            "event": "test",
            "detail": "PAN is ABCDE1234F",
            "raw": "Aadhaar: 1111 2222 3333",
        }
        result = redact_pii(None, "info", event_dict)
        assert REDACTED_PAN in result["detail"]
        assert REDACTED_AADHAAR in result["raw"]

    def test_preserves_non_string_fields(self):
        """Non-string fields are not modified."""
        event_dict = {
            "event": "test",
            "count": 42,
            "confidence": 0.95,
            "ready": True,
        }
        result = redact_pii(None, "info", event_dict)
        assert result["count"] == 42
        assert result["confidence"] == 0.95
        assert result["ready"] is True

    def test_returns_same_dict_reference(self):
        """Processor modifies event_dict in place and returns it."""
        event_dict = {"event": "test"}
        result = redact_pii(None, "info", event_dict)
        assert result is event_dict

    def test_mixed_pii_and_non_pii(self):
        """Dict with mixed PII and non-PII fields handles correctly."""
        event_dict = {
            "event": "Processing complete",
            "request_id": "req_abc123",
            "document_info": "Aadhaar 9876 5432 1098 found",
            "processing_time_ms": 150,
        }
        result = redact_pii(None, "info", event_dict)
        assert result["event"] == "Processing complete"
        assert result["request_id"] == "req_abc123"
        assert REDACTED_AADHAAR in result["document_info"]
        assert result["processing_time_ms"] == 150


class TestRequestLoggingMiddleware:
    """Tests for request logging middleware integration."""

    @pytest.mark.asyncio
    async def test_middleware_adds_request_id_header(self, async_client):
        """Middleware adds X-Request-ID response header."""
        response = await async_client.get("/health")
        assert "x-request-id" in response.headers
        assert response.headers["x-request-id"].startswith("req_")

    @pytest.mark.asyncio
    async def test_middleware_request_id_is_unique(self, async_client):
        """Each request gets a unique request_id."""
        response1 = await async_client.get("/health")
        response2 = await async_client.get("/health")
        assert response1.headers["x-request-id"] != response2.headers["x-request-id"]

    @pytest.mark.asyncio
    async def test_middleware_works_on_all_endpoints(self, async_client):
        """Middleware applies to all endpoints."""
        for endpoint in ["/health", "/ready"]:
            response = await async_client.get(endpoint)
            assert "x-request-id" in response.headers
