"""Unit tests for _check_pdf_safety function in app.validators.security module."""

import io
import zlib

import pytest

from app.validators.security import (
    MaliciousFileError,
    SecurityConfig,
    _check_pdf_safety,
)


def _create_minimal_pdf(num_pages: int = 1) -> bytes:
    """Create a minimal valid PDF with the specified number of pages.

    Uses pypdf's PdfWriter to generate a structurally valid PDF
    for testing purposes.
    """
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class TestCheckPdfSafetyNormalPdf:
    """Tests for _check_pdf_safety with valid/normal PDFs."""

    def test_single_page_pdf_passes(self) -> None:
        """A valid single-page PDF should pass all safety checks."""
        pdf_bytes = _create_minimal_pdf(num_pages=1)
        config = SecurityConfig(max_pdf_pages=10)
        # Should not raise
        result = _check_pdf_safety(pdf_bytes, config)
        assert result is None

    def test_multi_page_pdf_within_limit_passes(self) -> None:
        """A valid multi-page PDF within the page limit should pass."""
        pdf_bytes = _create_minimal_pdf(num_pages=5)
        config = SecurityConfig(max_pdf_pages=10)
        result = _check_pdf_safety(pdf_bytes, config)
        assert result is None

    def test_pdf_at_exact_page_limit_passes(self) -> None:
        """A PDF with exactly max_pdf_pages should pass."""
        pdf_bytes = _create_minimal_pdf(num_pages=10)
        config = SecurityConfig(max_pdf_pages=10)
        result = _check_pdf_safety(pdf_bytes, config)
        assert result is None


class TestCheckPdfSafetyPageCount:
    """Tests for _check_pdf_safety page count enforcement."""

    def test_pdf_exceeding_page_limit_raises(self) -> None:
        """A PDF with more pages than max_pdf_pages should raise MaliciousFileError."""
        pdf_bytes = _create_minimal_pdf(num_pages=11)
        config = SecurityConfig(max_pdf_pages=10)
        with pytest.raises(MaliciousFileError) as exc_info:
            _check_pdf_safety(pdf_bytes, config)
        assert "11 pages" in exc_info.value.reason
        assert "10" in exc_info.value.reason
        assert exc_info.value.error_code == "MALICIOUS_FILE"

    def test_pdf_with_many_pages_raises(self) -> None:
        """A PDF with significantly more pages than limit should raise."""
        pdf_bytes = _create_minimal_pdf(num_pages=20)
        config = SecurityConfig(max_pdf_pages=5)
        with pytest.raises(MaliciousFileError) as exc_info:
            _check_pdf_safety(pdf_bytes, config)
        assert "20 pages" in exc_info.value.reason

    def test_custom_page_limit_enforced(self) -> None:
        """Custom max_pdf_pages values are correctly enforced."""
        pdf_bytes = _create_minimal_pdf(num_pages=3)
        config = SecurityConfig(max_pdf_pages=2)
        with pytest.raises(MaliciousFileError):
            _check_pdf_safety(pdf_bytes, config)


class TestCheckPdfSafetyCorruptPdf:
    """Tests for _check_pdf_safety with corrupt/invalid PDFs."""

    def test_non_pdf_content_raises(self) -> None:
        """Non-PDF content that cannot be parsed should raise MaliciousFileError."""
        non_pdf_bytes = b"This is not a PDF at all, just random text content."
        config = SecurityConfig(max_pdf_pages=10)
        with pytest.raises(MaliciousFileError) as exc_info:
            _check_pdf_safety(non_pdf_bytes, config)
        assert "cannot be parsed" in exc_info.value.reason.lower() or "corrupt" in exc_info.value.reason.lower()
        assert exc_info.value.error_code == "MALICIOUS_FILE"

    def test_empty_bytes_raises(self) -> None:
        """Empty bytes should raise MaliciousFileError (can't parse)."""
        config = SecurityConfig(max_pdf_pages=10)
        with pytest.raises(MaliciousFileError) as exc_info:
            _check_pdf_safety(b"", config)
        assert exc_info.value.error_code == "MALICIOUS_FILE"

    def test_truncated_pdf_header_raises(self) -> None:
        """A truncated PDF with only header bytes should raise MaliciousFileError."""
        truncated = b"%PDF-1.7\n"
        config = SecurityConfig(max_pdf_pages=10)
        with pytest.raises(MaliciousFileError) as exc_info:
            _check_pdf_safety(truncated, config)
        assert exc_info.value.error_code == "MALICIOUS_FILE"

    def test_corrupted_pdf_body_raises(self) -> None:
        """A PDF with valid header but corrupted body should raise."""
        corrupted = b"%PDF-1.7\n" + b"\x00" * 100 + b"%%EOF"
        config = SecurityConfig(max_pdf_pages=10)
        with pytest.raises(MaliciousFileError) as exc_info:
            _check_pdf_safety(corrupted, config)
        assert exc_info.value.error_code == "MALICIOUS_FILE"

    def test_random_binary_raises(self) -> None:
        """Random binary data should raise MaliciousFileError."""
        import os as _os
        random_bytes = _os.urandom(1024)
        config = SecurityConfig(max_pdf_pages=10)
        with pytest.raises(MaliciousFileError):
            _check_pdf_safety(random_bytes, config)
