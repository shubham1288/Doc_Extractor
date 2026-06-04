"""Tests for API endpoints: health, readiness, and basic extraction flow."""

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    async def test_health_returns_200(self, async_client: AsyncClient):
        """Health endpoint should return HTTP 200."""
        response = await async_client.get("/health")
        assert response.status_code == 200

    async def test_health_returns_status_field(self, async_client: AsyncClient):
        """Health response should contain a 'status' field."""
        response = await async_client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    async def test_health_returns_ocr_ready_field(self, async_client: AsyncClient):
        """Health response should contain an 'ocr_ready' boolean field."""
        response = await async_client.get("/health")
        data = response.json()
        assert "ocr_ready" in data
        assert isinstance(data["ocr_ready"], bool)

    async def test_health_returns_timestamp(self, async_client: AsyncClient):
        """Health response should contain a 'timestamp' field."""
        response = await async_client.get("/health")
        data = response.json()
        assert "timestamp" in data


class TestReadyEndpoint:
    """Tests for GET /ready endpoint."""

    async def test_ready_returns_200(self, async_client: AsyncClient):
        """Ready endpoint should return HTTP 200."""
        response = await async_client.get("/ready")
        assert response.status_code == 200

    async def test_ready_returns_ready_field(self, async_client: AsyncClient):
        """Ready response should contain a 'ready' boolean field."""
        response = await async_client.get("/ready")
        data = response.json()
        assert "ready" in data
        assert isinstance(data["ready"], bool)

    async def test_ready_returns_checks(self, async_client: AsyncClient):
        """Ready response should contain 'checks' with ocr_model and memory."""
        response = await async_client.get("/ready")
        data = response.json()
        assert "checks" in data
        assert "ocr_model" in data["checks"]
        assert "memory" in data["checks"]

    async def test_ready_ocr_check_has_ready_field(self, async_client: AsyncClient):
        """OCR model check should have a 'ready' boolean."""
        response = await async_client.get("/ready")
        data = response.json()
        ocr_check = data["checks"]["ocr_model"]
        assert "ready" in ocr_check
        assert isinstance(ocr_check["ready"], bool)

    async def test_ready_memory_check_has_percent(self, async_client: AsyncClient):
        """Memory check should have percent_used field."""
        response = await async_client.get("/ready")
        data = response.json()
        memory_check = data["checks"]["memory"]
        assert "percent_used" in memory_check
        assert isinstance(memory_check["percent_used"], (int, float))
        assert 0 <= memory_check["percent_used"] <= 100

    async def test_ready_returns_timestamp(self, async_client: AsyncClient):
        """Ready response should contain a 'timestamp' field."""
        response = await async_client.get("/ready")
        data = response.json()
        assert "timestamp" in data


class TestExtractEndpoint:
    """Tests for POST /api/v1/kyc/extract endpoint."""

    async def test_extract_no_file_returns_422(self, async_client: AsyncClient):
        """Extract endpoint should return 422 when no file is provided."""
        response = await async_client.post("/api/v1/kyc/extract")
        assert response.status_code == 422

    async def test_extract_invalid_file_type_returns_400(self, async_client: AsyncClient):
        """Extract endpoint should return 400 for unsupported file types."""
        # Create a file with invalid magic bytes (plain text)
        file_content = b"This is not an image file at all, just plain text content."
        response = await async_client.post(
            "/api/v1/kyc/extract",
            files={"file": ("test.jpg", file_content, "image/jpeg")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error_code" in data
        assert data["error_code"] == "INVALID_FILE_TYPE"

    async def test_extract_file_too_large_returns_413(self, async_client: AsyncClient):
        """Extract endpoint should return 413 for files exceeding size limit."""
        # Create a valid JPEG header but with content exceeding 10MB
        jpeg_header = b"\xff\xd8\xff\xe0"
        large_content = jpeg_header + b"\x00" * (11 * 1024 * 1024)
        response = await async_client.post(
            "/api/v1/kyc/extract",
            files={"file": ("large.jpg", large_content, "image/jpeg")},
        )
        assert response.status_code == 413
        data = response.json()
        assert data["error_code"] == "FILE_TOO_LARGE"

    async def test_extract_error_response_has_request_id(self, async_client: AsyncClient):
        """Error responses should include a request_id."""
        file_content = b"not an image"
        response = await async_client.post(
            "/api/v1/kyc/extract",
            files={"file": ("test.jpg", file_content, "image/jpeg")},
        )
        data = response.json()
        assert "request_id" in data
        assert data["request_id"].startswith("req_")

    async def test_extract_error_response_has_timestamp(self, async_client: AsyncClient):
        """Error responses should include a timestamp."""
        file_content = b"not an image"
        response = await async_client.post(
            "/api/v1/kyc/extract",
            files={"file": ("test.jpg", file_content, "image/jpeg")},
        )
        data = response.json()
        assert "timestamp" in data
