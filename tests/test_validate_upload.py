"""Unit tests for the validate_upload() orchestration function."""

import io
import struct
import zlib

import pytest

from app.validators.security import (
    AllowedMimeType,
    FileTooLargeError,
    InvalidFileTypeError,
    SecurityConfig,
    ValidatedFile,
    validate_upload,
)


# ---------------------------------------------------------------------------
# Helpers to generate minimal valid file bytes
# ---------------------------------------------------------------------------


def _make_jpeg_bytes(size_padding: int = 0) -> bytes:
    """Create minimal valid JPEG file bytes."""
    # Minimal JPEG: SOI marker + APP0 marker + padding + EOI
    header = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.\' ',#\x1c\x1c(7),01444\x1f\'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\x9e\xa7\x8b"
    )
    padding = b"\x00" * size_padding
    tail = b"\xff\xd9"
    return header + padding + tail


def _make_png_bytes(size_padding: int = 0) -> bytes:
    """Create minimal valid PNG file bytes."""
    buf = io.BytesIO()
    buf.write(b"\x89PNG\r\n\x1a\n")

    def write_chunk(chunk_type: bytes, data: bytes) -> None:
        buf.write(struct.pack(">I", len(data)))
        buf.write(chunk_type)
        buf.write(data)
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        buf.write(struct.pack(">I", crc))

    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    write_chunk(b"IHDR", ihdr_data)

    raw_data = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw_data)
    write_chunk(b"IDAT", compressed + b"\x00" * size_padding)

    write_chunk(b"IEND", b"")
    return buf.getvalue()


def _make_pdf_bytes(num_pages: int = 1) -> bytes:
    """Create a valid PDF file with the given number of pages using pypdf."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateUploadJPEG:
    """Test validate_upload with valid JPEG files."""

    def test_valid_jpeg_returns_validated_file(self) -> None:
        """Valid JPEG bytes should pass all checks and return correct ValidatedFile."""
        jpeg_bytes = _make_jpeg_bytes()
        result = validate_upload(jpeg_bytes, "photo.jpg", "image/jpeg")

        assert isinstance(result, ValidatedFile)
        assert result.file_bytes is jpeg_bytes
        assert result.filename == "photo.jpg"
        assert result.mime_type == "image/jpeg"
        assert result.file_size == len(jpeg_bytes)
        assert result.is_pdf is False

    def test_valid_jpeg_with_jpeg_extension(self) -> None:
        """JPEG with .jpeg extension works."""
        jpeg_bytes = _make_jpeg_bytes()
        result = validate_upload(jpeg_bytes, "scan.jpeg", "image/jpeg")
        assert result.filename == "scan.jpeg"
        assert result.mime_type == "image/jpeg"


class TestValidateUploadPNG:
    """Test validate_upload with valid PNG files."""

    def test_valid_png_returns_validated_file(self) -> None:
        """Valid PNG bytes should pass all checks and return correct ValidatedFile."""
        png_bytes = _make_png_bytes()
        result = validate_upload(png_bytes, "document.png", "image/png")

        assert isinstance(result, ValidatedFile)
        assert result.file_bytes is png_bytes
        assert result.filename == "document.png"
        assert result.mime_type == "image/png"
        assert result.file_size == len(png_bytes)
        assert result.is_pdf is False


class TestValidateUploadPDF:
    """Test validate_upload with valid PDF files."""

    def test_valid_pdf_returns_validated_file_with_is_pdf_true(self) -> None:
        """Valid PDF bytes should pass and return ValidatedFile with is_pdf=True."""
        pdf_bytes = _make_pdf_bytes(num_pages=1)
        result = validate_upload(pdf_bytes, "document.pdf", "application/pdf")

        assert isinstance(result, ValidatedFile)
        assert result.file_bytes is pdf_bytes
        assert result.filename == "document.pdf"
        assert result.mime_type == "application/pdf"
        assert result.file_size == len(pdf_bytes)
        assert result.is_pdf is True

    def test_multi_page_pdf_under_limit_passes(self) -> None:
        """PDF with pages under the limit passes."""
        pdf_bytes = _make_pdf_bytes(num_pages=5)
        config = SecurityConfig(max_pdf_pages=10)
        result = validate_upload(pdf_bytes, "multi.pdf", "application/pdf", config=config)
        assert result.is_pdf is True


class TestValidateUploadFileTooLarge:
    """Test that oversized files are rejected with FileTooLargeError."""

    def test_too_large_file_raises_error(self) -> None:
        """File exceeding max size should raise FileTooLargeError."""
        # Use a 1MB limit config and create a file slightly over
        config = SecurityConfig(max_file_size_mb=1)
        # Create JPEG bytes that exceed 1MB
        large_bytes = _make_jpeg_bytes(size_padding=1024 * 1024 + 100)

        with pytest.raises(FileTooLargeError) as exc_info:
            validate_upload(large_bytes, "big.jpg", "image/jpeg", config=config)

        assert exc_info.value.file_size == len(large_bytes)
        assert exc_info.value.max_size == 1 * 1024 * 1024
        assert exc_info.value.error_code == "FILE_TOO_LARGE"


class TestValidateUploadInvalidFileType:
    """Test that unknown file types are rejected with InvalidFileTypeError."""

    def test_unknown_magic_bytes_rejected(self) -> None:
        """File with unrecognized magic bytes should raise InvalidFileTypeError."""
        unknown_bytes = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c" * 5

        with pytest.raises(InvalidFileTypeError) as exc_info:
            validate_upload(unknown_bytes, "mystery.jpg", "image/jpeg")

        assert exc_info.value.detected_type == "unknown"
        assert exc_info.value.error_code == "INVALID_FILE_TYPE"

    def test_too_short_file_rejected(self) -> None:
        """Very short file (< 12 bytes) should raise InvalidFileTypeError."""
        short_bytes = b"\xff\xd8\xff"  # Only 3 bytes

        with pytest.raises(InvalidFileTypeError):
            validate_upload(short_bytes, "tiny.jpg", "image/jpeg")


class TestValidateUploadContentTypeMismatchIgnored:
    """Test that the declared content_type is IGNORED — magic bytes win."""

    def test_wrong_content_type_ignored_for_jpeg(self) -> None:
        """JPEG bytes with wrong content_type still pass (magic bytes win)."""
        jpeg_bytes = _make_jpeg_bytes()
        # Declare content_type as PNG, but magic bytes say JPEG
        result = validate_upload(jpeg_bytes, "photo.jpg", "image/png")

        assert result.mime_type == "image/jpeg"  # magic bytes win
        assert result.is_pdf is False

    def test_wrong_content_type_ignored_for_png(self) -> None:
        """PNG bytes with wrong content_type still pass (magic bytes win)."""
        png_bytes = _make_png_bytes()
        # Declare content_type as PDF, but magic bytes say PNG
        result = validate_upload(png_bytes, "image.png", "application/pdf")

        assert result.mime_type == "image/png"  # magic bytes win
        assert result.is_pdf is False

    def test_wrong_content_type_ignored_for_pdf(self) -> None:
        """PDF bytes with wrong content_type still pass (magic bytes win)."""
        pdf_bytes = _make_pdf_bytes()
        # Declare content_type as JPEG, but magic bytes say PDF
        result = validate_upload(pdf_bytes, "file.pdf", "image/jpeg")

        assert result.mime_type == "application/pdf"  # magic bytes win
        assert result.is_pdf is True

    def test_garbage_content_type_ignored(self) -> None:
        """Completely invalid content_type is ignored when magic bytes are valid."""
        jpeg_bytes = _make_jpeg_bytes()
        result = validate_upload(jpeg_bytes, "photo.jpg", "totally/invalid-type")

        assert result.mime_type == "image/jpeg"


class TestValidateUploadDefaultConfig:
    """Test that a default SecurityConfig is created when none provided."""

    def test_no_config_uses_default(self) -> None:
        """When config is None, default SecurityConfig is used."""
        jpeg_bytes = _make_jpeg_bytes()
        result = validate_upload(jpeg_bytes, "photo.jpg", "image/jpeg", config=None)
        assert isinstance(result, ValidatedFile)

    def test_custom_config_respected(self) -> None:
        """Custom config parameters are respected."""
        config = SecurityConfig(max_file_size_mb=1)
        jpeg_bytes = _make_jpeg_bytes()
        # Small file should pass with custom config
        result = validate_upload(jpeg_bytes, "photo.jpg", "image/jpeg", config=config)
        assert isinstance(result, ValidatedFile)


class TestValidateUploadFilenameSanitization:
    """Test that the filename is sanitized in the returned ValidatedFile."""

    def test_filename_is_sanitized(self) -> None:
        """Path traversal in filename is stripped."""
        jpeg_bytes = _make_jpeg_bytes()
        result = validate_upload(jpeg_bytes, "../../etc/photo.jpg", "image/jpeg")

        assert ".." not in result.filename
        assert "/" not in result.filename
        assert result.filename.endswith(".jpg")

    def test_special_chars_removed_from_filename(self) -> None:
        """Special characters are removed from filename."""
        jpeg_bytes = _make_jpeg_bytes()
        result = validate_upload(jpeg_bytes, "my photo (1).jpg", "image/jpeg")

        assert " " not in result.filename
        assert "(" not in result.filename
        assert ")" not in result.filename
