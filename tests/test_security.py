"""Unit tests for app.validators.security module."""

import pytest

from app.validators.security import (
    AllowedMimeType,
    FileTooLargeError,
    InvalidFileTypeError,
    MaliciousFileError,
    SecurityConfig,
    SecurityValidationError,
    ValidatedFile,
    _check_file_size,
    _check_magic_bytes,
    _sanitize_filename,
    _validate_mime_type,
)


class TestAllowedMimeType:
    """Tests for AllowedMimeType enum."""

    def test_enum_values(self) -> None:
        assert AllowedMimeType.JPEG == "image/jpeg"
        assert AllowedMimeType.PNG == "image/png"
        assert AllowedMimeType.TIFF == "image/tiff"
        assert AllowedMimeType.WEBP == "image/webp"
        assert AllowedMimeType.PDF == "application/pdf"

    def test_enum_count(self) -> None:
        assert len(AllowedMimeType) == 5

    def test_enum_is_str(self) -> None:
        for mime in AllowedMimeType:
            assert isinstance(mime, str)
            assert isinstance(mime.value, str)


class TestSecurityConfig:
    """Tests for SecurityConfig dataclass."""

    def test_default_values(self) -> None:
        config = SecurityConfig()
        assert config.max_file_size_mb == 10
        assert config.max_pdf_pages == 10

    def test_default_extensions(self) -> None:
        config = SecurityConfig()
        expected = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".webp", ".pdf"})
        assert config.allowed_extensions == expected

    def test_default_mime_types(self) -> None:
        config = SecurityConfig()
        expected = frozenset(
            {"image/jpeg", "image/png", "image/tiff", "image/webp", "application/pdf"}
        )
        assert config.allowed_mime_types == expected

    def test_max_file_size_bytes_property(self) -> None:
        config = SecurityConfig()
        assert config.max_file_size_bytes == 10 * 1024 * 1024

    def test_custom_max_file_size(self) -> None:
        config = SecurityConfig(max_file_size_mb=5)
        assert config.max_file_size_mb == 5
        assert config.max_file_size_bytes == 5 * 1024 * 1024

    def test_custom_max_pdf_pages(self) -> None:
        config = SecurityConfig(max_pdf_pages=20)
        assert config.max_pdf_pages == 20

    def test_frozen_dataclass(self) -> None:
        config = SecurityConfig()
        with pytest.raises(Exception):
            config.max_file_size_mb = 20  # type: ignore[misc]

    def test_allowed_extensions_is_frozenset(self) -> None:
        config = SecurityConfig()
        assert isinstance(config.allowed_extensions, frozenset)

    def test_allowed_mime_types_is_frozenset(self) -> None:
        config = SecurityConfig()
        assert isinstance(config.allowed_mime_types, frozenset)


class TestValidatedFile:
    """Tests for ValidatedFile dataclass."""

    def test_creation(self) -> None:
        vf = ValidatedFile(
            file_bytes=b"test data",
            filename="document.pdf",
            mime_type="application/pdf",
            file_size=9,
            is_pdf=True,
        )
        assert vf.file_bytes == b"test data"
        assert vf.filename == "document.pdf"
        assert vf.mime_type == "application/pdf"
        assert vf.file_size == 9
        assert vf.is_pdf is True

    def test_image_file(self) -> None:
        vf = ValidatedFile(
            file_bytes=b"\xff\xd8\xff",
            filename="photo.jpg",
            mime_type="image/jpeg",
            file_size=3,
            is_pdf=False,
        )
        assert vf.is_pdf is False
        assert vf.mime_type == "image/jpeg"


class TestFileTooLargeError:
    """Tests for FileTooLargeError exception."""

    def test_attributes(self) -> None:
        err = FileTooLargeError(file_size=15_000_000, max_size=10_485_760)
        assert err.file_size == 15_000_000
        assert err.max_size == 10_485_760
        assert err.error_code == "FILE_TOO_LARGE"
        assert "15000000" in err.message
        assert "10485760" in err.message

    def test_is_security_validation_error(self) -> None:
        err = FileTooLargeError(file_size=1, max_size=0)
        assert isinstance(err, SecurityValidationError)

    def test_is_exception(self) -> None:
        err = FileTooLargeError(file_size=1, max_size=0)
        assert isinstance(err, Exception)


class TestInvalidFileTypeError:
    """Tests for InvalidFileTypeError exception."""

    def test_attributes(self) -> None:
        err = InvalidFileTypeError(
            detected_type="application/x-executable",
            allowed_types=frozenset({"image/jpeg", "image/png"}),
        )
        assert err.detected_type == "application/x-executable"
        assert err.error_code == "INVALID_FILE_TYPE"
        assert "application/x-executable" in err.message

    def test_without_allowed_types(self) -> None:
        err = InvalidFileTypeError(detected_type="text/html")
        assert err.detected_type == "text/html"
        assert err.allowed_types is None

    def test_is_security_validation_error(self) -> None:
        err = InvalidFileTypeError(detected_type="text/plain")
        assert isinstance(err, SecurityValidationError)


class TestMaliciousFileError:
    """Tests for MaliciousFileError exception."""

    def test_attributes(self) -> None:
        err = MaliciousFileError(reason="PDF bomb detected: excessive page count")
        assert err.reason == "PDF bomb detected: excessive page count"
        assert err.error_code == "MALICIOUS_FILE"
        assert "PDF bomb" in err.message

    def test_is_security_validation_error(self) -> None:
        err = MaliciousFileError(reason="test")
        assert isinstance(err, SecurityValidationError)


class TestCheckFileSize:
    """Tests for _check_file_size function."""

    def test_file_exactly_at_limit_passes(self) -> None:
        config = SecurityConfig(max_file_size_mb=1)
        file_bytes = b"\x00" * (1 * 1024 * 1024)  # exactly 1MB
        # Should not raise
        assert _check_file_size(file_bytes, config) is None

    def test_file_one_byte_over_limit_raises(self) -> None:
        config = SecurityConfig(max_file_size_mb=1)
        file_bytes = b"\x00" * (1 * 1024 * 1024 + 1)  # 1MB + 1 byte
        with pytest.raises(FileTooLargeError) as exc_info:
            _check_file_size(file_bytes, config)
        assert exc_info.value.file_size == 1 * 1024 * 1024 + 1
        assert exc_info.value.max_size == 1 * 1024 * 1024

    def test_empty_file_passes(self) -> None:
        config = SecurityConfig(max_file_size_mb=10)
        # Empty file should pass size validation
        assert _check_file_size(b"", config) is None

    def test_normal_file_under_limit_passes(self) -> None:
        config = SecurityConfig(max_file_size_mb=10)
        file_bytes = b"\x00" * 1024  # 1KB, well under 10MB limit
        assert _check_file_size(file_bytes, config) is None


class TestCheckMagicBytes:
    """Tests for _check_magic_bytes function."""

    def test_detect_jpeg(self) -> None:
        # JPEG magic bytes: FF D8 FF followed by arbitrary content
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        assert _check_magic_bytes(jpeg_bytes) == "image/jpeg"

    def test_detect_png(self) -> None:
        # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        assert _check_magic_bytes(png_bytes) == "image/png"

    def test_detect_tiff_little_endian(self) -> None:
        # TIFF LE: 49 49 2A 00
        tiff_le_bytes = b"II*\x00" + b"\x00" * 20
        assert _check_magic_bytes(tiff_le_bytes) == "image/tiff"

    def test_detect_tiff_big_endian(self) -> None:
        # TIFF BE: 4D 4D 00 2A
        tiff_be_bytes = b"MM\x00*" + b"\x00" * 20
        assert _check_magic_bytes(tiff_be_bytes) == "image/tiff"

    def test_detect_webp(self) -> None:
        # WEBP: RIFF at offset 0, file size at 4-7, WEBP at offset 8
        webp_bytes = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 20
        assert _check_magic_bytes(webp_bytes) == "image/webp"

    def test_detect_pdf(self) -> None:
        # PDF: %PDF
        pdf_bytes = b"%PDF-1.7" + b"\x00" * 20
        assert _check_magic_bytes(pdf_bytes) == "application/pdf"

    def test_unknown_file_type_raises_error(self) -> None:
        # Random bytes that don't match any signature
        unknown_bytes = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d"
        with pytest.raises(InvalidFileTypeError) as exc_info:
            _check_magic_bytes(unknown_bytes)
        assert exc_info.value.detected_type == "unknown"
        assert exc_info.value.error_code == "INVALID_FILE_TYPE"

    def test_too_short_bytes_raises_error(self) -> None:
        # Files shorter than 12 bytes cannot be reliably identified
        short_bytes = b"\xff\xd8"
        with pytest.raises(InvalidFileTypeError) as exc_info:
            _check_magic_bytes(short_bytes)
        assert exc_info.value.detected_type == "unknown"

    def test_empty_bytes_raises_error(self) -> None:
        with pytest.raises(InvalidFileTypeError):
            _check_magic_bytes(b"")

    def test_riff_without_webp_raises_error(self) -> None:
        # RIFF header but not WEBP (e.g., AVI or WAV)
        riff_avi_bytes = b"RIFF" + b"\x00\x00\x00\x00" + b"AVI " + b"\x00" * 20
        with pytest.raises(InvalidFileTypeError):
            _check_magic_bytes(riff_avi_bytes)

    def test_allowed_types_in_error(self) -> None:
        unknown_bytes = b"\x00" * 20
        with pytest.raises(InvalidFileTypeError) as exc_info:
            _check_magic_bytes(unknown_bytes)
        assert exc_info.value.allowed_types is not None
        assert "image/jpeg" in exc_info.value.allowed_types
        assert "image/png" in exc_info.value.allowed_types
        assert "application/pdf" in exc_info.value.allowed_types


class TestValidateMimeType:
    """Tests for _validate_mime_type function."""

    @pytest.mark.parametrize(
        "mime_type",
        [
            "image/jpeg",
            "image/png",
            "image/tiff",
            "image/webp",
            "application/pdf",
        ],
    )
    def test_all_allowed_mime_types_pass(self, mime_type: str) -> None:
        """All 5 allowed MIME types should pass validation without raising."""
        config = SecurityConfig()
        assert _validate_mime_type(mime_type, config) is None

    @pytest.mark.parametrize(
        "mime_type",
        [
            "application/x-executable",
            "text/html",
            "application/javascript",
            "image/svg+xml",
            "application/zip",
            "text/plain",
        ],
    )
    def test_disallowed_mime_types_raise_error(self, mime_type: str) -> None:
        """Unknown or disallowed MIME types should raise InvalidFileTypeError."""
        config = SecurityConfig()
        with pytest.raises(InvalidFileTypeError) as exc_info:
            _validate_mime_type(mime_type, config)
        assert exc_info.value.detected_type == mime_type
        assert exc_info.value.error_code == "INVALID_FILE_TYPE"

    def test_error_includes_full_set_of_allowed_types(self) -> None:
        """The raised error should include all allowed MIME types."""
        config = SecurityConfig()
        with pytest.raises(InvalidFileTypeError) as exc_info:
            _validate_mime_type("text/html", config)
        assert exc_info.value.allowed_types == config.allowed_mime_types
        # Verify all 5 types are present
        assert "image/jpeg" in exc_info.value.allowed_types
        assert "image/png" in exc_info.value.allowed_types
        assert "image/tiff" in exc_info.value.allowed_types
        assert "image/webp" in exc_info.value.allowed_types
        assert "application/pdf" in exc_info.value.allowed_types

    def test_custom_allowed_types(self) -> None:
        """Validation should respect custom allowed_mime_types in config."""
        config = SecurityConfig(
            allowed_mime_types=frozenset({"image/jpeg", "image/png"})
        )
        # Allowed types pass
        assert _validate_mime_type("image/jpeg", config) is None
        assert _validate_mime_type("image/png", config) is None
        # Normally-allowed types now rejected
        with pytest.raises(InvalidFileTypeError):
            _validate_mime_type("application/pdf", config)


class TestSanitizeFilename:
    """Tests for _sanitize_filename function."""

    def test_normal_filename_passes_through(self) -> None:
        """A simple valid filename should pass through unchanged."""
        assert _sanitize_filename("document.pdf") == "document.pdf"

    def test_normal_jpeg_filename(self) -> None:
        """Normal JPEG filenames pass through correctly."""
        assert _sanitize_filename("photo.jpg") == "photo.jpg"
        assert _sanitize_filename("my-scan.jpeg") == "my-scan.jpeg"

    def test_normal_png_filename(self) -> None:
        """Normal PNG filename passes through correctly."""
        assert _sanitize_filename("id_card.png") == "id_card.png"

    def test_normal_tiff_filename(self) -> None:
        """Normal TIFF filename passes through correctly."""
        assert _sanitize_filename("scan.tiff") == "scan.tiff"

    def test_normal_webp_filename(self) -> None:
        """Normal WEBP filename passes through correctly."""
        assert _sanitize_filename("image.webp") == "image.webp"

    def test_path_traversal_unix_stripped(self) -> None:
        """Unix path traversal attempts are stripped."""
        result = _sanitize_filename("../../etc/passwd.pdf")
        # After basename extraction and sanitization, path components are removed
        assert "/" not in result
        assert ".." not in result
        # The result should have .pdf extension and be valid
        assert result.endswith(".pdf")

    def test_path_traversal_deep_unix(self) -> None:
        """Deep Unix path traversal is handled."""
        result = _sanitize_filename("../../../secret/file.png")
        assert "/" not in result
        assert "\\" not in result
        assert ".." not in result
        assert result.endswith(".png")

    def test_windows_path_separators_handled(self) -> None:
        """Windows backslash path separators are handled."""
        result = _sanitize_filename("C:\\Users\\attacker\\malware.pdf")
        assert "\\" not in result
        assert result.endswith(".pdf")

    def test_windows_path_traversal(self) -> None:
        """Windows-style path traversal is stripped."""
        result = _sanitize_filename("..\\..\\windows\\system32\\config.pdf")
        assert "\\" not in result
        assert ".." not in result
        assert result.endswith(".pdf")

    def test_null_bytes_removed(self) -> None:
        """Null bytes are stripped from filename."""
        result = _sanitize_filename("document\x00.pdf")
        assert "\x00" not in result
        assert result.endswith(".pdf")

    def test_null_bytes_in_middle(self) -> None:
        """Null bytes in the middle of the filename are removed."""
        result = _sanitize_filename("doc\x00ument\x00.jpg")
        assert "\x00" not in result
        assert result == "document.jpg"

    def test_empty_filename_returns_default(self) -> None:
        """Empty filename after sanitization returns default name."""
        result = _sanitize_filename("...")
        assert result == "unnamed_document"

    def test_only_special_chars_returns_default(self) -> None:
        """Filename with only special characters returns default."""
        result = _sanitize_filename("@#$%^&*()")
        assert result == "unnamed_document"

    def test_only_dots_returns_default(self) -> None:
        """Filename that is only dots returns default."""
        result = _sanitize_filename("....")
        assert result == "unnamed_document"

    def test_special_characters_removed(self) -> None:
        """Special characters are removed from filenames."""
        result = _sanitize_filename("my file (1).pdf")
        # Spaces and parentheses are removed
        assert " " not in result
        assert "(" not in result
        assert ")" not in result
        assert result == "myfile1.pdf"

    def test_unicode_characters_removed(self) -> None:
        """Unicode/non-ASCII characters are removed."""
        result = _sanitize_filename("dôcümënt.pdf")
        assert result == "dcmnt.pdf"

    def test_preserves_hyphens_underscores(self) -> None:
        """Hyphens and underscores are preserved."""
        assert _sanitize_filename("my-document_v2.pdf") == "my-document_v2.pdf"

    def test_extension_validation_allowed(self) -> None:
        """Valid extensions from SecurityConfig pass validation."""
        config = SecurityConfig()
        for ext in [".jpg", ".jpeg", ".png", ".tiff", ".webp", ".pdf"]:
            filename = f"document{ext}"
            result = _sanitize_filename(filename, config)
            assert result == filename

    def test_extension_validation_rejected(self) -> None:
        """Disallowed extensions raise InvalidFileTypeError."""
        config = SecurityConfig()
        with pytest.raises(InvalidFileTypeError) as exc_info:
            _sanitize_filename("malware.exe", config)
        assert exc_info.value.detected_type == ".exe"
        assert exc_info.value.error_code == "INVALID_FILE_TYPE"

    def test_extension_validation_rejected_html(self) -> None:
        """HTML extension is rejected."""
        with pytest.raises(InvalidFileTypeError):
            _sanitize_filename("page.html")

    def test_extension_validation_rejected_js(self) -> None:
        """JavaScript extension is rejected."""
        with pytest.raises(InvalidFileTypeError):
            _sanitize_filename("script.js")

    def test_no_extension_raises_error(self) -> None:
        """Filename with no extension raises InvalidFileTypeError."""
        with pytest.raises(InvalidFileTypeError) as exc_info:
            _sanitize_filename("document")
        assert exc_info.value.detected_type == "no_extension"

    def test_case_insensitive_extension(self) -> None:
        """Extension validation is case-insensitive."""
        result = _sanitize_filename("document.PDF")
        assert result == "document.PDF"

    def test_mixed_path_and_null_bytes(self) -> None:
        """Combination of path traversal and null bytes is handled."""
        result = _sanitize_filename("../../\x00secret.pdf")
        assert "/" not in result
        assert "\x00" not in result
        assert ".." not in result
        assert result.endswith(".pdf")

    def test_leading_trailing_whitespace_stripped(self) -> None:
        """Leading and trailing whitespace is stripped."""
        result = _sanitize_filename("  document.pdf  ")
        assert result == "document.pdf"

    def test_leading_dots_stripped(self) -> None:
        """Leading dots are stripped (prevents hidden files)."""
        result = _sanitize_filename(".hidden.pdf")
        assert not result.startswith(".")
        assert result == "hidden.pdf"
