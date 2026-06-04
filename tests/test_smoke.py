"""
Smoke tests to verify the test infrastructure works correctly.

Validates:
- pytest collection and execution
- pytest-asyncio async test support
- hypothesis property testing framework
- FastAPI test client fixture
- numpy image fixtures
- File bytes fixtures
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st


class TestPytestInfrastructure:
    """Basic pytest functionality."""

    def test_assert_true(self):
        """Simplest possible test to verify pytest runs."""
        assert True

    def test_imports_work(self):
        """Verify core project imports are functional."""
        from app.main import app
        assert app is not None
        assert app.title == "KYC Document Extraction API"


class TestAsyncSupport:
    """Verify pytest-asyncio works with auto mode."""

    async def test_async_function(self):
        """Async test should run without explicit markers (asyncio_mode=auto)."""
        result = await _async_identity(42)
        assert result == 42

    async def test_async_client_fixture(self, async_client):
        """FastAPI test client fixture should be usable in async tests."""
        # A basic request to the app - even if no routes match,
        # we verify the client is functional
        response = await async_client.get("/")
        # FastAPI returns 404 for undefined routes
        assert response.status_code in (200, 404)


class TestHypothesis:
    """Verify hypothesis property testing framework."""

    @given(st.integers(min_value=0, max_value=1000))
    def test_integers_are_non_negative(self, n):
        """Property: generated integers within range are non-negative."""
        assert n >= 0

    @given(st.text(min_size=1, max_size=50))
    def test_text_is_non_empty(self, s):
        """Property: generated text with min_size=1 is always non-empty."""
        assert len(s) >= 1

    @given(st.binary(min_size=1, max_size=100))
    def test_binary_has_correct_type(self, b):
        """Property: generated binary data is always bytes."""
        assert isinstance(b, bytes)
        assert len(b) >= 1


class TestImageFixtures:
    """Verify numpy image fixtures work correctly."""

    def test_grayscale_image_shape(self, sample_grayscale_image):
        """Grayscale fixture should be 2D with correct dimensions."""
        assert sample_grayscale_image.shape == (100, 100)
        assert sample_grayscale_image.dtype == np.uint8

    def test_color_image_shape(self, sample_color_image):
        """Color fixture should be 3D (H, W, 3) with correct dimensions."""
        assert sample_color_image.shape == (100, 100, 3)
        assert sample_color_image.dtype == np.uint8

    def test_document_image_shape(self, sample_document_image):
        """Document fixture should be grayscale with text-like regions."""
        assert sample_document_image.shape == (300, 200)
        assert sample_document_image.dtype == np.uint8
        # Should have both light (background) and dark (text) regions
        assert sample_document_image.min() < 50
        assert sample_document_image.max() > 200


class TestFileFixtures:
    """Verify file bytes fixtures are valid."""

    def test_jpeg_bytes_start_with_magic(self, valid_jpeg_bytes):
        """JPEG fixture should start with FF D8 FF magic bytes."""
        assert valid_jpeg_bytes[:2] == b"\xff\xd8"
        assert len(valid_jpeg_bytes) > 10

    def test_png_bytes_start_with_magic(self, valid_png_bytes):
        """PNG fixture should start with PNG signature."""
        assert valid_png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(valid_png_bytes) > 10

    def test_invalid_bytes_not_matching_formats(self, invalid_file_bytes):
        """Invalid fixture should not match JPEG or PNG magic bytes."""
        assert invalid_file_bytes[:2] != b"\xff\xd8"
        assert invalid_file_bytes[:8] != b"\x89PNG\r\n\x1a\n"


class TestMarkers:
    """Verify custom markers are registered."""

    @pytest.mark.slow
    def test_slow_marker_works(self):
        """The 'slow' marker should be registered without warnings."""
        assert True

    @pytest.mark.integration
    def test_integration_marker_works(self):
        """The 'integration' marker should be registered without warnings."""
        assert True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _async_identity(value):
    """Simple async helper for testing."""
    return value
