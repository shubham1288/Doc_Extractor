"""
Property-based tests for Security Validator — file size enforcement.

Uses Hypothesis to verify that the file size invariant holds for ALL
generated inputs:
- For any file_bytes where len(file_bytes) > max_size_bytes, _check_file_size()
  must raise FileTooLargeError
- For any file_bytes where len(file_bytes) <= max_size_bytes, _check_file_size()
  must NOT raise

**Validates: Requirements 1.1**
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.validators.security import (
    FileTooLargeError,
    SecurityConfig,
    _check_file_size,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Random max_file_size_mb values between 1 and 50
max_file_size_mb_strategy = st.integers(min_value=1, max_value=50)


class TestFileSizeEnforcementProperty:
    """Property 1: File size enforcement.

    For any file_bytes where len(file_bytes) > max_size_bytes,
    _check_file_size() must raise FileTooLargeError.

    For any file_bytes where len(file_bytes) <= max_size_bytes,
    _check_file_size() must NOT raise.

    **Validates: Requirements 1.1**
    """

    @given(
        max_file_size_mb=max_file_size_mb_strategy,
        extra_bytes=st.integers(min_value=1, max_value=1024),
    )
    @settings(max_examples=100, deadline=500)
    def test_oversized_files_always_rejected(
        self, max_file_size_mb: int, extra_bytes: int
    ) -> None:
        """For any file exceeding the limit, FileTooLargeError must be raised."""
        config = SecurityConfig(max_file_size_mb=max_file_size_mb)
        max_size_bytes = max_file_size_mb * 1024 * 1024

        # Generate file bytes that exceed the limit by extra_bytes
        file_bytes = b"\x00" * (max_size_bytes + extra_bytes)

        with pytest.raises(FileTooLargeError) as exc_info:
            _check_file_size(file_bytes, config)

        # Verify the error contains correct metadata
        assert exc_info.value.file_size == len(file_bytes)
        assert exc_info.value.max_size == max_size_bytes

    @given(
        max_file_size_mb=max_file_size_mb_strategy,
        file_size_fraction=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100, deadline=500)
    def test_undersized_files_always_accepted(
        self, max_file_size_mb: int, file_size_fraction: float
    ) -> None:
        """For any file within the limit, no exception must be raised."""
        config = SecurityConfig(max_file_size_mb=max_file_size_mb)
        max_size_bytes = max_file_size_mb * 1024 * 1024

        # Generate file bytes that are at or under the limit
        file_size = int(max_size_bytes * file_size_fraction)
        file_bytes = b"\x00" * file_size

        # Must NOT raise any exception
        result = _check_file_size(file_bytes, config)
        assert result is None

    @given(max_file_size_mb=max_file_size_mb_strategy)
    @settings(max_examples=100, deadline=500)
    def test_file_exactly_at_limit_accepted(self, max_file_size_mb: int) -> None:
        """For any config, a file exactly at the byte limit must be accepted."""
        config = SecurityConfig(max_file_size_mb=max_file_size_mb)
        max_size_bytes = max_file_size_mb * 1024 * 1024

        # Exactly at the limit
        file_bytes = b"\x00" * max_size_bytes

        # Must NOT raise
        result = _check_file_size(file_bytes, config)
        assert result is None

    @given(max_file_size_mb=max_file_size_mb_strategy)
    @settings(max_examples=100, deadline=500)
    def test_file_one_byte_over_limit_rejected(self, max_file_size_mb: int) -> None:
        """For any config, a file one byte over the limit must be rejected."""
        config = SecurityConfig(max_file_size_mb=max_file_size_mb)
        max_size_bytes = max_file_size_mb * 1024 * 1024

        # One byte over the limit
        file_bytes = b"\x00" * (max_size_bytes + 1)

        with pytest.raises(FileTooLargeError) as exc_info:
            _check_file_size(file_bytes, config)

        assert exc_info.value.file_size == max_size_bytes + 1
        assert exc_info.value.max_size == max_size_bytes

    @given(
        max_file_size_mb=max_file_size_mb_strategy,
        file_content=st.binary(min_size=0, max_size=1024),
    )
    @settings(max_examples=100, deadline=500)
    def test_small_random_bytes_always_accepted(
        self, max_file_size_mb: int, file_content: bytes
    ) -> None:
        """Any small random byte sequence (<=1KB) is always under the MB limit."""
        config = SecurityConfig(max_file_size_mb=max_file_size_mb)

        # Small files (max 1KB) are always under any MB-level limit (min 1MB)
        result = _check_file_size(file_content, config)
        assert result is None


# ---------------------------------------------------------------------------
# Property 2: Magic bytes always override declared content type
# ---------------------------------------------------------------------------

from app.validators.security import validate_upload, ValidatedFile, InvalidFileTypeError


# Minimal valid file content for each magic byte signature
_JPEG_MAGIC = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG with JFIF marker
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # PNG signature

# A minimal valid PDF that pypdf can parse without raising MaliciousFileError
_PDF_MAGIC = (
    b"%PDF-1.0\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n183\n%%EOF"
)

# Map from magic bytes to expected MIME type
_MAGIC_TO_MIME: dict[str, tuple[bytes, str]] = {
    "jpeg": (_JPEG_MAGIC, "image/jpeg"),
    "png": (_PNG_MAGIC, "image/png"),
    "pdf": (_PDF_MAGIC, "application/pdf"),
}

# Strategies for generating random content_type strings
_random_content_type_strategy = st.one_of(
    # Valid MIME types that differ from actual magic bytes
    st.sampled_from([
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
        "application/pdf",
        "application/octet-stream",
        "text/plain",
        "text/html",
        "application/json",
        "application/xml",
        "image/gif",
        "video/mp4",
        "audio/mpeg",
    ]),
    # Garbage/invalid content type strings
    st.text(min_size=1, max_size=50).filter(lambda s: "/" in s or len(s) > 0),
)

# Strategy that picks one of the known magic byte file types
_magic_file_strategy = st.sampled_from(["jpeg", "png", "pdf"])


class TestMagicBytesOverrideProperty:
    """Property 2: Magic bytes always override declared content type.

    For any valid file bytes (JPEG/PNG/PDF magic), validate_upload() must
    return a MIME type matching the magic bytes, regardless of what
    content_type is declared. The content_type parameter must NEVER
    influence the detected MIME type in the result.

    **Validates: Requirements 1.2**
    """

    @given(
        file_type=_magic_file_strategy,
        declared_content_type=_random_content_type_strategy,
    )
    @settings(max_examples=200, deadline=2000)
    def test_magic_bytes_determine_mime_type_not_declared_content_type(
        self, file_type: str, declared_content_type: str
    ) -> None:
        """For any valid magic bytes, result.mime_type matches magic, not declared content_type."""
        file_bytes, expected_mime = _MAGIC_TO_MIME[file_type]

        # Choose a matching filename extension for the magic bytes type
        # (so extension validation passes)
        extension_map = {"jpeg": "test.jpg", "png": "test.png", "pdf": "test.pdf"}
        filename = extension_map[file_type]

        config = SecurityConfig(max_file_size_mb=10)
        result = validate_upload(
            file_bytes=file_bytes,
            filename=filename,
            content_type=declared_content_type,
            config=config,
        )

        # The detected MIME type MUST match the magic bytes, NOT the declared content_type
        assert result.mime_type == expected_mime, (
            f"Expected mime_type={expected_mime} (from magic bytes), "
            f"but got {result.mime_type}. Declared content_type was '{declared_content_type}'"
        )

    @given(
        file_type=_magic_file_strategy,
        garbage_content_type=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
            min_size=1,
            max_size=100,
        ),
    )
    @settings(max_examples=200, deadline=2000)
    def test_garbage_content_type_never_influences_detection(
        self, file_type: str, garbage_content_type: str
    ) -> None:
        """Even with complete garbage as content_type, magic bytes still determine result."""
        file_bytes, expected_mime = _MAGIC_TO_MIME[file_type]

        extension_map = {"jpeg": "test.jpg", "png": "test.png", "pdf": "test.pdf"}
        filename = extension_map[file_type]

        config = SecurityConfig(max_file_size_mb=10)
        result = validate_upload(
            file_bytes=file_bytes,
            filename=filename,
            content_type=garbage_content_type,
            config=config,
        )

        assert result.mime_type == expected_mime, (
            f"Garbage content_type '{garbage_content_type}' influenced detection. "
            f"Expected {expected_mime}, got {result.mime_type}"
        )

    @given(
        declared_content_type=_random_content_type_strategy,
        extra_bytes=st.binary(min_size=0, max_size=512),
    )
    @settings(max_examples=100, deadline=2000)
    def test_jpeg_magic_always_detected_as_jpeg(
        self, declared_content_type: str, extra_bytes: bytes
    ) -> None:
        """JPEG magic bytes always produce image/jpeg regardless of declared type."""
        file_bytes = _JPEG_MAGIC + extra_bytes

        config = SecurityConfig(max_file_size_mb=10)
        result = validate_upload(
            file_bytes=file_bytes,
            filename="photo.jpg",
            content_type=declared_content_type,
            config=config,
        )

        assert result.mime_type == "image/jpeg"

    @given(
        declared_content_type=_random_content_type_strategy,
        extra_bytes=st.binary(min_size=0, max_size=512),
    )
    @settings(max_examples=100, deadline=2000)
    def test_png_magic_always_detected_as_png(
        self, declared_content_type: str, extra_bytes: bytes
    ) -> None:
        """PNG magic bytes always produce image/png regardless of declared type."""
        file_bytes = _PNG_MAGIC + extra_bytes

        config = SecurityConfig(max_file_size_mb=10)
        result = validate_upload(
            file_bytes=file_bytes,
            filename="image.png",
            content_type=declared_content_type,
            config=config,
        )

        assert result.mime_type == "image/png"

    @given(
        declared_content_type=_random_content_type_strategy,
    )
    @settings(max_examples=100, deadline=2000)
    def test_pdf_magic_always_detected_as_pdf(
        self, declared_content_type: str
    ) -> None:
        """PDF magic bytes always produce application/pdf regardless of declared type."""
        # Use the minimal valid PDF as-is (cannot append random bytes to a PDF)
        file_bytes = _PDF_MAGIC

        config = SecurityConfig(max_file_size_mb=10)
        result = validate_upload(
            file_bytes=file_bytes,
            filename="document.pdf",
            content_type=declared_content_type,
            config=config,
        )

        assert result.mime_type == "application/pdf"


# ---------------------------------------------------------------------------
# Property 3: Disallowed extensions and MIME types are always rejected
# ---------------------------------------------------------------------------

from app.validators.security import _check_magic_bytes, _sanitize_filename


# Known magic byte prefixes for allowed MIME types — used to filter them out
_KNOWN_MAGIC_PREFIXES: list[bytes] = [
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",    # PNG
    b"II*\x00",              # TIFF little-endian
    b"MM\x00*",              # TIFF big-endian
    b"%PDF",                 # PDF
    b"RIFF",                 # WEBP starts with RIFF (combined with WEBP at offset 8)
]


def _starts_with_known_magic(data: bytes) -> bool:
    """Check if data starts with any known allowed magic byte prefix."""
    for prefix in _KNOWN_MAGIC_PREFIXES:
        if data[: len(prefix)] == prefix:
            return True
    return False


# Strategy: generate random bytes that do NOT start with any known magic prefix
_disallowed_file_bytes_strategy = st.binary(min_size=12, max_size=512).filter(
    lambda b: not _starts_with_known_magic(b)
)

# Strategy: generate filenames with disallowed extensions
_DISALLOWED_EXTENSIONS = [
    ".exe", ".bat", ".sh", ".html", ".js", ".py", ".rb", ".php",
    ".cmd", ".vbs", ".ps1", ".msi", ".dll", ".so", ".com",
    ".jar", ".war", ".class", ".cgi", ".pl", ".asp", ".aspx",
    ".svg", ".gif", ".bmp", ".ico", ".mp3", ".mp4", ".avi",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".zip", ".tar",
    ".gz", ".rar", ".7z", ".iso", ".dmg", ".apk", ".deb",
]

# Strategy for base filenames (simple alphanumeric names)
_base_filename_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
).filter(lambda s: len(s.strip().strip(".")) > 0)

_disallowed_extension_strategy = st.sampled_from(_DISALLOWED_EXTENSIONS)

# Combine filename base + disallowed extension
_disallowed_filename_strategy = st.builds(
    lambda base, ext: base + ext,
    _base_filename_strategy,
    _disallowed_extension_strategy,
)


class TestExtensionMimeRejectionProperty:
    """Property 3: Disallowed extensions and MIME types are always rejected.

    For any file_bytes whose magic bytes don't match an allowed MIME type,
    validate_upload() must raise InvalidFileTypeError.

    For any filename with a disallowed extension (not in .jpg, .jpeg, .png,
    .tiff, .webp, .pdf), _sanitize_filename() must raise InvalidFileTypeError.

    **Validates: Requirements 1.3, 1.6**
    """

    @given(file_bytes=_disallowed_file_bytes_strategy)
    @settings(max_examples=200, deadline=1000)
    def test_unrecognized_magic_bytes_always_rejected(
        self, file_bytes: bytes
    ) -> None:
        """For any bytes not matching a known magic signature, _check_magic_bytes raises InvalidFileTypeError."""
        with pytest.raises(InvalidFileTypeError) as exc_info:
            _check_magic_bytes(file_bytes)

        assert exc_info.value.error_code == "INVALID_FILE_TYPE"
        assert exc_info.value.detected_type == "unknown"

    @given(filename=_disallowed_filename_strategy)
    @settings(max_examples=200, deadline=1000)
    def test_disallowed_extension_always_rejected(
        self, filename: str
    ) -> None:
        """For any filename with a disallowed extension, _sanitize_filename raises InvalidFileTypeError."""
        config = SecurityConfig(max_file_size_mb=10)

        with pytest.raises(InvalidFileTypeError) as exc_info:
            _sanitize_filename(filename, config)

        assert exc_info.value.error_code == "INVALID_FILE_TYPE"

    @given(
        file_bytes=_disallowed_file_bytes_strategy,
        filename=_disallowed_filename_strategy,
        content_type=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=200, deadline=2000)
    def test_validate_upload_rejects_unrecognized_magic_bytes(
        self, file_bytes: bytes, filename: str, content_type: str
    ) -> None:
        """For any file with unrecognized magic bytes, validate_upload raises InvalidFileTypeError."""
        config = SecurityConfig(max_file_size_mb=10)

        with pytest.raises(InvalidFileTypeError) as exc_info:
            validate_upload(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
                config=config,
            )

        assert exc_info.value.error_code == "INVALID_FILE_TYPE"

    @given(
        extension=_disallowed_extension_strategy,
        base_name=st.just("document"),
    )
    @settings(max_examples=100, deadline=1000)
    def test_each_disallowed_extension_rejected(
        self, extension: str, base_name: str
    ) -> None:
        """Every disallowed extension in our list is rejected by _sanitize_filename."""
        config = SecurityConfig(max_file_size_mb=10)
        filename = base_name + extension

        with pytest.raises(InvalidFileTypeError) as exc_info:
            _sanitize_filename(filename, config)

        assert exc_info.value.error_code == "INVALID_FILE_TYPE"
        # The detected_type should be the extension (lowercased)
        assert exc_info.value.detected_type == extension.lower()

    @given(
        random_ext=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=200, deadline=1000)
    def test_random_extension_not_in_allowed_set_rejected(
        self, random_ext: str
    ) -> None:
        """For any random extension not in the allowed set, _sanitize_filename raises InvalidFileTypeError."""
        allowed = {".jpg", ".jpeg", ".png", ".tiff", ".webp", ".pdf"}
        ext_with_dot = "." + random_ext.lower()

        # Only test extensions that are NOT in the allowed set
        assume(ext_with_dot not in allowed)
        # Ensure the extension is non-empty after lowering
        assume(len(random_ext.strip()) > 0)

        config = SecurityConfig(max_file_size_mb=10)
        filename = "testfile." + random_ext

        with pytest.raises(InvalidFileTypeError) as exc_info:
            _sanitize_filename(filename, config)

        assert exc_info.value.error_code == "INVALID_FILE_TYPE"


# ---------------------------------------------------------------------------
# Property 4: Filename sanitization removes all dangerous characters
# ---------------------------------------------------------------------------


# Allowed extensions for generating valid filenames
_ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".webp", ".pdf"]

# Characters considered dangerous in filenames
_DANGEROUS_CHARS = "/\\\x00"
_DANGEROUS_SEQUENCE = ".."

# Strategy: generate random base strings with dangerous characters mixed in
_dangerous_char_alphabet = st.characters(
    whitelist_categories=("L", "N", "P", "S", "Cc"),
    whitelist_characters="/\\\x00.~!@#$%^&*()[]{}|;:'\",<>?`",
)

_dangerous_base_strategy = st.text(
    alphabet=_dangerous_char_alphabet,
    min_size=0,
    max_size=100,
)

# Strategy: specifically inject path traversal patterns
_path_traversal_patterns = st.sampled_from([
    "../", "..\\", "../../", "..\\..\\",
    "/etc/passwd", "\\windows\\system32",
    "..%2f", "\x00", "....//", "..../\\",
    "/", "\\", "../../../", "..\\..\\..\\",
])

# Strategy for allowed extensions
_allowed_extension_strategy = st.sampled_from(_ALLOWED_EXTENSIONS)


class TestFilenameSanitizationProperty:
    """Property 4: Filename sanitization removes all dangerous characters.

    For any input filename, the sanitized output must never contain:
    - Path separators (/ or \\)
    - Null bytes (\\x00)
    - Double-dots (..)

    For any input filename with an allowed extension, the sanitized output
    must contain only alphanumeric characters, dots, hyphens, and underscores.

    The sanitized filename is always non-empty (either sanitized input or
    "unnamed_document" default).

    **Validates: Requirements 1.5**
    """

    @given(
        base=_dangerous_base_strategy,
        extension=_allowed_extension_strategy,
    )
    @settings(max_examples=200, deadline=1000)
    def test_no_path_separators_in_output(
        self, base: str, extension: str
    ) -> None:
        """For any input filename, the sanitized output never contains path separators."""
        filename = base + extension
        config = SecurityConfig(max_file_size_mb=10)

        try:
            result = _sanitize_filename(filename, config)
        except InvalidFileTypeError:
            # If the extension gets mangled during sanitization,
            # the function may raise — that's acceptable behavior
            return

        assert "/" not in result, (
            f"Sanitized filename contains '/': '{result}' (input: '{filename}')"
        )
        assert "\\" not in result, (
            f"Sanitized filename contains '\\': '{result}' (input: '{filename}')"
        )

    @given(
        base=_dangerous_base_strategy,
        extension=_allowed_extension_strategy,
    )
    @settings(max_examples=200, deadline=1000)
    def test_no_null_bytes_in_output(
        self, base: str, extension: str
    ) -> None:
        """For any input filename, the sanitized output never contains null bytes."""
        filename = base + extension
        config = SecurityConfig(max_file_size_mb=10)

        try:
            result = _sanitize_filename(filename, config)
        except InvalidFileTypeError:
            return

        assert "\x00" not in result, (
            f"Sanitized filename contains null byte: '{result!r}' (input: '{filename!r}')"
        )

    @given(
        base=_dangerous_base_strategy,
        extension=_allowed_extension_strategy,
    )
    @settings(max_examples=200, deadline=1000)
    def test_no_double_dots_in_output(
        self, base: str, extension: str
    ) -> None:
        """For any input filename, the sanitized output never contains '..'."""
        filename = base + extension
        config = SecurityConfig(max_file_size_mb=10)

        try:
            result = _sanitize_filename(filename, config)
        except InvalidFileTypeError:
            return

        assert ".." not in result, (
            f"Sanitized filename contains '..': '{result}' (input: '{filename}')"
        )

    @given(
        traversal=_path_traversal_patterns,
        safe_base=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
            min_size=1,
            max_size=20,
        ),
        extension=_allowed_extension_strategy,
    )
    @settings(max_examples=200, deadline=1000)
    def test_path_traversal_patterns_removed(
        self, traversal: str, safe_base: str, extension: str
    ) -> None:
        """Explicit path traversal patterns are always removed from output."""
        filename = traversal + safe_base + extension
        config = SecurityConfig(max_file_size_mb=10)

        try:
            result = _sanitize_filename(filename, config)
        except InvalidFileTypeError:
            return

        assert "/" not in result
        assert "\\" not in result
        assert "\x00" not in result
        assert ".." not in result

    @given(
        base=_dangerous_base_strategy,
        extension=_allowed_extension_strategy,
    )
    @settings(max_examples=200, deadline=1000)
    def test_output_contains_only_safe_characters(
        self, base: str, extension: str
    ) -> None:
        """Sanitized output contains only alphanumeric chars, dots, hyphens, and underscores."""
        filename = base + extension
        config = SecurityConfig(max_file_size_mb=10)

        try:
            result = _sanitize_filename(filename, config)
        except InvalidFileTypeError:
            return

        import re
        safe_pattern = re.compile(r"^[a-zA-Z0-9.\-_]+$")
        assert safe_pattern.match(result), (
            f"Sanitized filename contains unsafe characters: '{result}' (input: '{filename!r}')"
        )

    @given(
        base=st.text(min_size=0, max_size=100),
        extension=_allowed_extension_strategy,
    )
    @settings(max_examples=200, deadline=1000)
    def test_sanitized_filename_always_non_empty(
        self, base: str, extension: str
    ) -> None:
        """The sanitized filename is always non-empty (returns default if needed)."""
        filename = base + extension
        config = SecurityConfig(max_file_size_mb=10)

        try:
            result = _sanitize_filename(filename, config)
        except InvalidFileTypeError:
            # If extension validation fails, the function raises rather than
            # returning empty — this is acceptable
            return

        assert len(result) > 0, (
            f"Sanitized filename is empty (input: '{filename!r}')"
        )

    @given(
        base=st.just(""),
        extension=_allowed_extension_strategy,
    )
    @settings(max_examples=50, deadline=1000)
    def test_empty_base_produces_default_or_valid(
        self, base: str, extension: str
    ) -> None:
        """An empty base with a valid extension produces a non-empty sanitized result."""
        filename = base + extension
        config = SecurityConfig(max_file_size_mb=10)

        try:
            result = _sanitize_filename(filename, config)
        except InvalidFileTypeError:
            return

        assert len(result) > 0

    @given(
        unicode_chars=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "M", "N", "P", "S", "Z", "C"),
            ),
            min_size=1,
            max_size=50,
        ),
        extension=_allowed_extension_strategy,
    )
    @settings(max_examples=200, deadline=1000)
    def test_unicode_and_control_chars_stripped(
        self, unicode_chars: str, extension: str
    ) -> None:
        """Unicode and control characters are stripped, leaving only safe chars."""
        filename = unicode_chars + extension
        config = SecurityConfig(max_file_size_mb=10)

        try:
            result = _sanitize_filename(filename, config)
        except InvalidFileTypeError:
            return

        import re
        safe_pattern = re.compile(r"^[a-zA-Z0-9.\-_]+$")
        assert safe_pattern.match(result), (
            f"Output contains non-safe chars: '{result!r}' (input: '{filename!r}')"
        )
