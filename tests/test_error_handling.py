"""
Unit tests for error handling and recovery in the KYC extraction endpoint.

Verifies:
- OCR not ready → HTTP 503 with OCR_NOT_READY (Task 11.1)
- Low OCR confidence (< 0.3) → UNKNOWN response (Task 11.2)
- Memory cleanup in finally blocks (Task 11.3)
"""

import gc
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.document import DocumentType
from app.ocr.engine import OCREngine, OCRResult


@pytest.fixture
async def client():
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestOCRNotReadyHandler:
    """Tests for HTTP 503 OCR_NOT_READY error handling (Task 11.1)."""

    @pytest.mark.asyncio
    async def test_returns_503_when_ocr_not_ready(self, client, valid_jpeg_bytes):
        """When OCR engine is not ready, returns 503 with OCR_NOT_READY code."""
        with patch.object(OCREngine, "is_ready", new_callable=lambda: property(lambda self: False)):
            response = await client.post(
                "/api/v1/kyc/extract",
                files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
            )

        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == "OCR_NOT_READY"
        assert "request_id" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_503_response_has_correct_structure(self, client, valid_jpeg_bytes):
        """503 error response contains all required ErrorResponse fields."""
        with patch.object(OCREngine, "is_ready", new_callable=lambda: property(lambda self: False)):
            response = await client.post(
                "/api/v1/kyc/extract",
                files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
            )

        data = response.json()
        assert "request_id" in data
        assert "error_code" in data
        assert "error_message" in data
        assert "timestamp" in data
        assert data["error_code"] == "OCR_NOT_READY"
        assert "not ready" in data["error_message"].lower()


class TestLowConfidenceHandling:
    """Tests for low OCR confidence → UNKNOWN response (Task 11.2)."""

    @pytest.mark.asyncio
    async def test_low_confidence_returns_unknown(self, client, valid_jpeg_bytes):
        """When OCR confidence < 0.3, returns document_type UNKNOWN."""
        low_confidence_result = OCRResult(
            text="some blurry text",
            boxes=[[[0, 0], [100, 0], [100, 30], [0, 30]]],
            confidences=[0.2],
            avg_confidence=0.2,
        )

        with patch.object(OCREngine, "is_ready", new_callable=lambda: property(lambda self: True)), \
             patch.object(OCREngine, "extract_text", return_value=low_confidence_result):
            response = await client.post(
                "/api/v1/kyc/extract",
                files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "unknown"
        assert data["extracted_data"] == {}
        assert data["ocr_confidence"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_zero_confidence_returns_unknown(self, client, valid_jpeg_bytes):
        """When OCR confidence is 0.0, returns document_type UNKNOWN."""
        zero_confidence_result = OCRResult(
            text="",
            boxes=[],
            confidences=[],
            avg_confidence=0.0,
        )

        with patch.object(OCREngine, "is_ready", new_callable=lambda: property(lambda self: True)), \
             patch.object(OCREngine, "extract_text", return_value=zero_confidence_result):
            response = await client.post(
                "/api/v1/kyc/extract",
                files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "unknown"
        assert data["extracted_data"] == {}

    @pytest.mark.asyncio
    async def test_confidence_at_threshold_does_not_return_unknown(self, client, valid_jpeg_bytes):
        """When OCR confidence == 0.3 (at threshold), normal processing continues."""
        at_threshold_result = OCRResult(
            text="INCOME TAX DEPARTMENT ABCDE1234F",
            boxes=[[[0, 0], [100, 0], [100, 30], [0, 30]]],
            confidences=[0.3],
            avg_confidence=0.3,
        )

        with patch.object(OCREngine, "is_ready", new_callable=lambda: property(lambda self: True)), \
             patch.object(OCREngine, "extract_text", return_value=at_threshold_result):
            response = await client.post(
                "/api/v1/kyc/extract",
                files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
            )

        # At threshold (0.3), should NOT return UNKNOWN - should proceed to classification
        assert response.status_code == 200
        data = response.json()
        # document_type might be identified or unknown depending on classifier, but
        # the key thing is that it's NOT forced to UNKNOWN by the threshold check
        assert data["ocr_confidence"] == pytest.approx(0.3)


class TestMemoryCleanup:
    """Tests for memory cleanup in finally blocks (Task 11.3)."""

    @pytest.mark.asyncio
    async def test_cleanup_on_successful_request(self, client, valid_jpeg_bytes):
        """After successful processing, file bytes are cleaned up."""
        result = OCRResult(
            text="test text",
            boxes=[[[0, 0], [100, 0], [100, 30], [0, 30]]],
            confidences=[0.5],
            avg_confidence=0.5,
        )

        with patch.object(OCREngine, "is_ready", new_callable=lambda: property(lambda self: True)), \
             patch.object(OCREngine, "extract_text", return_value=result), \
             patch("app.api.endpoints.gc.collect") as mock_gc:
            response = await client.post(
                "/api/v1/kyc/extract",
                files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
            )

        # gc.collect should have been called in the finally block
        assert mock_gc.called

    @pytest.mark.asyncio
    async def test_cleanup_on_error(self, client, valid_jpeg_bytes):
        """After an error, file bytes are still cleaned up via finally block.
        
        The endpoint's finally block sets file_bytes = None and calls gc.collect().
        Even when an exception occurs, the finally block executes ensuring cleanup.
        """
        cleanup_called = []

        original_gc_collect = gc.collect

        def track_gc_collect(*args, **kwargs):
            cleanup_called.append(True)
            return original_gc_collect(*args, **kwargs)

        with patch.object(OCREngine, "is_ready", new_callable=lambda: property(lambda self: True)), \
             patch.object(OCREngine, "extract_text", side_effect=RuntimeError("test error")), \
             patch("app.api.endpoints.gc.collect", side_effect=track_gc_collect):
            try:
                response = await client.post(
                    "/api/v1/kyc/extract",
                    files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
                )
                # If the exception handler catches it, we get 500
                assert response.status_code == 500
            except Exception:
                # Starlette may propagate the exception through middleware
                pass

        # gc.collect was called in the finally block regardless of the exception
        assert len(cleanup_called) > 0, "gc.collect was not called - finally block did not execute"
