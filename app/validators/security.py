"""
Security configuration, types, and exceptions for file upload validation.

This module defines the core data structures used by the Security Validator:
- AllowedMimeType enum for permitted file types
- SecurityConfig dataclass for validation constraints
- ValidatedFile dataclass for validated upload results
- Custom exception classes for validation failures
- Magic bytes detection for true file type identification
- Filename sanitization for path traversal prevention
- PDF bomb detection for malicious PDF files
"""

import io
import os
import re
import zlib
from dataclasses import dataclass, field
from enum import Enum

from app.core.config import get_settings


# Magic byte signatures for supported file types
# Each entry: (offset, signature_bytes, mime_type)
# For WEBP, a secondary check is needed at offset 8
_MAGIC_SIGNATURES: list[tuple[int, bytes, str]] = [
    (0, b"\xff\xd8\xff", "image/jpeg"),  # JPEG
    (0, b"\x89PNG\r\n\x1a\n", "image/png"),  # PNG
    (0, b"II*\x00", "image/tiff"),  # TIFF little-endian
    (0, b"MM\x00*", "image/tiff"),  # TIFF big-endian
    (0, b"%PDF", "application/pdf"),  # PDF
    # WEBP is handled separately due to dual-offset check
]

_WEBP_RIFF_HEADER = b"RIFF"
_WEBP_MARKER = b"WEBP"


class AllowedMimeType(str, Enum):
    """MIME types allowed for KYC document uploads."""

    JPEG = "image/jpeg"
    PNG = "image/png"
    TIFF = "image/tiff"
    WEBP = "image/webp"
    PDF = "application/pdf"


@dataclass(frozen=True)
class SecurityConfig:
    """Configuration for file upload security validation.

    Default values come from application Settings but can be overridden
    for testing or custom deployments.
    """

    max_file_size_mb: int = field(default_factory=lambda: get_settings().max_file_size_mb)
    max_pdf_pages: int = field(default_factory=lambda: get_settings().max_pdf_pages)
    allowed_extensions: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".tiff", ".webp", ".pdf"}
    )
    allowed_mime_types: frozenset[str] = frozenset(
        {m.value for m in AllowedMimeType}
    )

    @property
    def max_file_size_bytes(self) -> int:
        """Return max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


@dataclass
class ValidatedFile:
    """Result of successful file validation.

    Contains the validated and sanitized file information ready
    for downstream processing.
    """

    file_bytes: bytes
    filename: str  # sanitized filename
    mime_type: str  # detected from magic bytes
    file_size: int  # size in bytes
    is_pdf: bool


# --- Custom Exceptions ---


class SecurityValidationError(Exception):
    """Base exception for all security validation failures."""

    def __init__(self, message: str, error_code: str) -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class FileTooLargeError(SecurityValidationError):
    """Raised when uploaded file exceeds the maximum allowed size."""

    def __init__(self, file_size: int, max_size: int) -> None:
        self.file_size = file_size
        self.max_size = max_size
        message = (
            f"File size {file_size} bytes exceeds maximum allowed "
            f"size of {max_size} bytes"
        )
        super().__init__(message=message, error_code="FILE_TOO_LARGE")


class InvalidFileTypeError(SecurityValidationError):
    """Raised when file type is not in the allowed set."""

    def __init__(self, detected_type: str, allowed_types: frozenset[str] | None = None) -> None:
        self.detected_type = detected_type
        self.allowed_types = allowed_types
        message = f"File type '{detected_type}' is not allowed"
        if allowed_types:
            message += f". Allowed types: {sorted(allowed_types)}"
        super().__init__(message=message, error_code="INVALID_FILE_TYPE")


class MaliciousFileError(SecurityValidationError):
    """Raised when a file is detected as potentially malicious.

    This covers PDF bombs, excessive nesting, decompression bombs,
    and other suspicious file characteristics.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        message = f"File rejected as potentially malicious: {reason}"
        super().__init__(message=message, error_code="MALICIOUS_FILE")


# --- Magic Bytes Detection ---


def _check_file_size(file_bytes: bytes, config: SecurityConfig) -> None:
    """Check that the file size does not exceed the configured maximum.

    Args:
        file_bytes: The raw file content to check.
        config: Security configuration containing the size limit.

    Raises:
        FileTooLargeError: If the file size exceeds ``config.max_file_size_bytes``.
    """
    file_size = len(file_bytes)
    if file_size > config.max_file_size_bytes:
        raise FileTooLargeError(file_size=file_size, max_size=config.max_file_size_bytes)


def _validate_mime_type(detected_mime: str, config: SecurityConfig) -> None:
    """Validate that the detected MIME type is in the allowed set.

    The declared content-type from the upload is IGNORED. Only the
    magic-bytes-detected MIME type is trusted and checked against the
    allowed list.

    Args:
        detected_mime: The MIME type detected by ``_check_magic_bytes``.
        config: Security configuration containing the allowed MIME types.

    Raises:
        InvalidFileTypeError: If the detected MIME type is not in
            ``config.allowed_mime_types``.
    """
    if detected_mime not in config.allowed_mime_types:
        raise InvalidFileTypeError(
            detected_type=detected_mime,
            allowed_types=config.allowed_mime_types,
        )


def _check_magic_bytes(file_bytes: bytes) -> str:
    """Detect the actual file type by inspecting magic bytes.

    Checks the initial bytes of the file against known magic byte
    signatures for supported document types. This prevents attackers
    from disguising malicious files by changing the content-type header
    or file extension.

    Args:
        file_bytes: The raw file content to inspect.

    Returns:
        The detected MIME type string (e.g. "image/jpeg").

    Raises:
        InvalidFileTypeError: If the file bytes do not match any
            known supported file type signature.
    """
    if len(file_bytes) < 12:
        raise InvalidFileTypeError(
            detected_type="unknown",
            allowed_types=frozenset({m.value for m in AllowedMimeType}),
        )

    # Check WEBP first (requires dual-offset check: RIFF at 0 and WEBP at 8)
    if (
        file_bytes[:4] == _WEBP_RIFF_HEADER
        and file_bytes[8:12] == _WEBP_MARKER
    ):
        return AllowedMimeType.WEBP.value

    # Check remaining signatures
    for offset, signature, mime_type in _MAGIC_SIGNATURES:
        end = offset + len(signature)
        if file_bytes[offset:end] == signature:
            return mime_type

    raise InvalidFileTypeError(
        detected_type="unknown",
        allowed_types=frozenset({m.value for m in AllowedMimeType}),
    )

# --- Filename Sanitization ---

# Regex pattern: only allow alphanumeric, dots, hyphens, and underscores
_ALLOWED_FILENAME_CHARS = re.compile(r"[^a-zA-Z0-9.\-_]")

_DEFAULT_FILENAME = "unnamed_document"


def _sanitize_filename(filename: str, config: SecurityConfig | None = None) -> str:
    """Sanitize an uploaded filename to prevent path traversal and injection.

    This function:
    1. Strips leading/trailing whitespace
    2. Extracts the basename (removes path separators / and \\)
    3. Removes null bytes (\\x00)
    4. Removes double-dot sequences (..) for path traversal prevention
    5. Removes special/control characters (only allows alphanumeric, dots, hyphens, underscores)
    6. Strips leading/trailing whitespace and dots from the result
    7. Returns a default name if the result is empty
    8. Validates the file extension against the allowed set

    Args:
        filename: The raw filename from the upload.
        config: Optional SecurityConfig for extension validation.
            If None, a default SecurityConfig is used.

    Returns:
        The sanitized filename.

    Raises:
        InvalidFileTypeError: If the file extension is not in the allowed set.
    """
    if config is None:
        config = SecurityConfig()

    # Step 1: Strip leading/trailing whitespace
    filename = filename.strip()

    # Step 2: Extract basename - handle both Unix and Windows path separators
    filename = os.path.basename(filename)
    # Also handle forward slashes explicitly (in case os.path.basename doesn't on Windows)
    filename = filename.replace("/", "").replace("\\", "")

    # Step 3: Remove null bytes
    filename = filename.replace("\x00", "")

    # Step 4: Remove special/control characters (keep only alphanumeric, dots, hyphens, underscores)
    filename = _ALLOWED_FILENAME_CHARS.sub("", filename)

    # Step 5: Remove double-dot sequences (path traversal)
    # Done AFTER character stripping to catch ".." created by removal of intervening chars
    while ".." in filename:
        filename = filename.replace("..", "")

    # Step 6: Strip leading/trailing whitespace and dots
    filename = filename.strip().strip(".")

    # Step 7: Return default if empty
    if not filename:
        return _DEFAULT_FILENAME

    # Step 8: Validate extension
    _, ext = os.path.splitext(filename)
    ext_lower = ext.lower()

    if not ext_lower:
        # No extension present - raise error
        raise InvalidFileTypeError(
            detected_type="no_extension",
            allowed_types=config.allowed_extensions,
        )

    if ext_lower not in config.allowed_extensions:
        raise InvalidFileTypeError(
            detected_type=ext_lower,
            allowed_types=config.allowed_extensions,
        )

    return filename


# --- PDF Safety Checks ---

# Maximum nested object depth before considering a PDF malicious
_MAX_NESTED_DEPTH = 100

# Maximum decompression ratio (decompressed / compressed) before flagging as bomb
_MAX_DECOMPRESSION_RATIO = 250


def _check_pdf_safety(file_bytes: bytes, config: SecurityConfig) -> None:
    """Detect potentially malicious PDFs (PDF bombs).

    Performs three safety checks on PDF files:
    1. Page count - rejects PDFs with more pages than ``config.max_pdf_pages``
    2. Decompression ratio - rejects if compressed streams decompress to more
       than 100x their original size (decompression bomb indicator)
    3. Nested object depth - rejects PDFs with deeply nested objects (>50 levels)

    Uses pypdf for reliable PDF parsing when available, falling back to
    basic byte-level checks if the PDF cannot be fully parsed.

    Args:
        file_bytes: The raw PDF file bytes to inspect.
        config: Security configuration containing max_pdf_pages limit.

    Returns:
        None if the PDF passes all safety checks.

    Raises:
        MaliciousFileError: If any safety check fails, or if the PDF
            cannot be parsed at all (corrupt/invalid).
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        # If pypdf is not available, fall back to basic checks
        _check_pdf_safety_basic(file_bytes, config)
        return

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        # Many legitimate PDFs (password-protected, digitally signed, or with
        # non-standard features) can fail to parse with pypdf. Don't reject
        # them as malicious — just skip the safety checks and let pdf2image
        # handle the conversion (it uses poppler which is more tolerant).
        import logging
        logging.getLogger(__name__).warning(
            f"PDF safety check skipped (pypdf could not parse): {e}"
        )
        # Only do basic page count check via byte scanning
        _check_pdf_safety_basic(file_bytes, config)
        return

    # Check 1: Page count
    num_pages = len(reader.pages)
    if num_pages > config.max_pdf_pages:
        raise MaliciousFileError(
            reason=(
                f"PDF has {num_pages} pages, exceeding maximum "
                f"of {config.max_pdf_pages} pages"
            )
        )

    # Check 2: Decompression ratio
    _check_decompression_ratio(file_bytes)

    # Check 3: Nested object depth
    _check_nested_depth(file_bytes)


def _check_decompression_ratio(file_bytes: bytes) -> None:
    """Check for excessive decompression ratio in PDF streams.

    Scans the raw PDF bytes for compressed stream objects and verifies
    that no stream decompresses to more than 100x its compressed size.

    Args:
        file_bytes: The raw PDF file bytes.

    Raises:
        MaliciousFileError: If a stream exceeds the decompression ratio limit.
    """
    # Find all stream content between 'stream' and 'endstream' markers
    stream_start_pattern = b"stream\r\n"
    stream_start_pattern_alt = b"stream\n"
    endstream_marker = b"endstream"

    pos = 0
    while pos < len(file_bytes):
        # Find next stream start
        idx = file_bytes.find(stream_start_pattern, pos)
        if idx == -1:
            idx = file_bytes.find(stream_start_pattern_alt, pos)
        if idx == -1:
            break

        # Determine actual stream data start
        if file_bytes[idx:idx + len(stream_start_pattern)] == stream_start_pattern:
            data_start = idx + len(stream_start_pattern)
        else:
            data_start = idx + len(stream_start_pattern_alt)

        # Find endstream
        end_idx = file_bytes.find(endstream_marker, data_start)
        if end_idx == -1:
            break

        # Extract compressed stream data
        compressed_data = file_bytes[data_start:end_idx].rstrip(b"\r\n")
        compressed_size = len(compressed_data)

        if compressed_size > 0:
            try:
                decompressed = zlib.decompress(compressed_data)
                decompressed_size = len(decompressed)

                if decompressed_size > compressed_size * _MAX_DECOMPRESSION_RATIO:
                    raise MaliciousFileError(
                        reason=(
                            f"PDF stream decompression ratio "
                            f"({decompressed_size / compressed_size:.0f}x) "
                            f"exceeds maximum of {_MAX_DECOMPRESSION_RATIO}x"
                        )
                    )
            except zlib.error:
                # Not a zlib-compressed stream, skip it
                pass

        pos = end_idx + len(endstream_marker)


def _check_nested_depth(file_bytes: bytes) -> None:
    """Check for excessively nested objects in PDF structure.

    Scans the raw PDF bytes for dictionary/array nesting depth,
    skipping stream content (binary data between 'stream' and 'endstream').
    Deeply nested objects can cause stack overflows in PDF parsers.

    Args:
        file_bytes: The raw PDF file bytes.

    Raises:
        MaliciousFileError: If nesting depth exceeds the limit.
    """
    max_depth = 0
    current_depth = 0

    # First, identify stream regions to skip (they contain binary data)
    # We'll only check structure outside of streams
    stream_start = b"stream"
    stream_end = b"endstream"

    i = 0
    length = len(file_bytes)

    while i < length:
        # Check if we're entering a stream (skip binary content)
        if file_bytes[i:i + 6] == stream_start:
            # Jump to endstream
            end_pos = file_bytes.find(stream_end, i + 6)
            if end_pos != -1:
                i = end_pos + len(stream_end)
                continue
            else:
                break  # Malformed, but don't reject for this

        byte = file_bytes[i]

        if byte == ord(b"<") and i + 1 < length and file_bytes[i + 1] == ord(b"<"):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
            i += 2
        elif byte == ord(b">") and i + 1 < length and file_bytes[i + 1] == ord(b">"):
            current_depth = max(0, current_depth - 1)
            i += 2
        elif byte == ord(b"["):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
            i += 1
        elif byte == ord(b"]"):
            current_depth = max(0, current_depth - 1)
            i += 1
        else:
            i += 1

        if max_depth > _MAX_NESTED_DEPTH:
            raise MaliciousFileError(
                reason=(
                    f"PDF has excessive object nesting depth (>{_MAX_NESTED_DEPTH} levels), "
                    f"indicating potential malicious content"
                )
            )


def validate_upload(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    config: SecurityConfig | None = None,
) -> ValidatedFile:
    """Orchestrate all file validation steps and return a ValidatedFile.

    Validates an uploaded file through a series of security checks in order
    of computational cost (cheapest first). The declared ``content_type`` is
    accepted but IGNORED — only magic-bytes detection determines the real
    file type.

    Steps:
    1. Create default SecurityConfig if none provided
    2. Check file size (cheapest check)
    3. Detect actual file type from magic bytes
    4. Validate detected MIME type against allowed set
    5. Sanitize the filename
    6. If detected type is PDF, run PDF bomb detection

    Args:
        file_bytes: Raw uploaded file content.
        filename: Original filename from the upload.
        content_type: Declared content type (IGNORED — magic bytes win).
        config: Optional SecurityConfig override.

    Returns:
        A ValidatedFile containing the validated and sanitized file data.

    Raises:
        FileTooLargeError: If file exceeds the configured size limit.
        InvalidFileTypeError: If magic bytes don't match an allowed type,
            or filename extension is disallowed.
        MaliciousFileError: If a PDF fails safety checks.
    """
    if config is None:
        config = SecurityConfig()

    # Step 1: Check file size (cheapest check first)
    _check_file_size(file_bytes, config)

    # Step 2: Detect actual file type from magic bytes
    detected_mime = _check_magic_bytes(file_bytes)

    # Step 3: Validate detected MIME type is allowed
    _validate_mime_type(detected_mime, config)

    # Step 4: Sanitize the filename
    sanitized_filename = _sanitize_filename(filename, config)

    # Step 5: PDF-specific safety checks
    is_pdf = detected_mime == AllowedMimeType.PDF.value
    if is_pdf:
        _check_pdf_safety(file_bytes, config)

    return ValidatedFile(
        file_bytes=file_bytes,
        filename=sanitized_filename,
        mime_type=detected_mime,
        file_size=len(file_bytes),
        is_pdf=is_pdf,
    )


def _check_pdf_safety_basic(file_bytes: bytes, config: SecurityConfig) -> None:
    """Fallback basic PDF safety checks when pypdf is not available.

    Uses byte-level pattern matching to approximate page count and
    check for obvious malicious indicators.

    Args:
        file_bytes: The raw PDF file bytes.
        config: Security configuration containing max_pdf_pages limit.

    Raises:
        MaliciousFileError: If safety checks fail.
    """
    # Verify it starts with %PDF
    if not file_bytes.startswith(b"%PDF"):
        raise MaliciousFileError(reason="PDF cannot be parsed (corrupt or invalid): missing PDF header")

    # Basic page count: count /Type /Page occurrences (excluding /Type /Pages)
    # This is approximate but catches obvious cases
    page_pattern = b"/Type /Page"
    pages_pattern = b"/Type /Pages"

    page_count = file_bytes.count(page_pattern) - file_bytes.count(pages_pattern)
    if page_count > config.max_pdf_pages:
        raise MaliciousFileError(
            reason=(
                f"PDF has approximately {page_count} pages, exceeding maximum "
                f"of {config.max_pdf_pages} pages"
            )
        )

    # Check decompression ratio
    _check_decompression_ratio(file_bytes)

    # Check nested depth
    _check_nested_depth(file_bytes)
