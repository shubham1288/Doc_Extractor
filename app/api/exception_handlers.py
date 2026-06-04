"""
Global exception handlers for the KYC API.

Catches known security exceptions and unhandled errors, returning
structured ErrorResponse objects. Never exposes PII in error responses.
"""

import logging
import gc

from fastapi import Request
from fastapi.responses import JSONResponse

from app.schemas.responses import ErrorResponse
from app.utils.request_id import generate_request_id
from app.validators.security import (
    FileTooLargeError,
    InvalidFileTypeError,
    MaliciousFileError,
    SecurityValidationError,
)

logger = logging.getLogger(__name__)


async def security_validation_exception_handler(
    request: Request, exc: SecurityValidationError
) -> JSONResponse:
    """Handle SecurityValidationError and its subclasses.

    Maps specific security errors to appropriate HTTP status codes:
    - FileTooLargeError → 413 Payload Too Large
    - InvalidFileTypeError → 400 Bad Request
    - MaliciousFileError → 400 Bad Request
    - Other SecurityValidationError → 400 Bad Request

    Never includes PII in the error response.
    """
    request_id = getattr(request.state, "request_id", generate_request_id())

    if isinstance(exc, FileTooLargeError):
        status_code = 413
        error_message = "File size exceeds the maximum allowed limit"
    elif isinstance(exc, InvalidFileTypeError):
        status_code = 400
        error_message = "File type is not supported"
    elif isinstance(exc, MaliciousFileError):
        status_code = 400
        error_message = "File rejected for security reasons"
    else:
        status_code = 400
        error_message = "File validation failed"

    logger.warning(
        "Security validation failed",
        extra={
            "request_id": request_id,
            "error_code": exc.error_code,
            "status_code": status_code,
        },
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error_code=exc.error_code,
        error_message=error_message,
    )

    # Force cleanup of any file data that may be in memory
    gc.collect()

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(mode="json"),
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle any unhandled exception.

    Returns HTTP 500 with a generic error message. Never exposes
    internal details or PII in the response. Logs the error for
    debugging (without PII).
    """
    request_id = getattr(request.state, "request_id", generate_request_id())

    logger.error(
        "Unhandled exception during request processing",
        extra={
            "request_id": request_id,
            "error_type": type(exc).__name__,
        },
        exc_info=True,
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error_code="INTERNAL_ERROR",
        error_message="An internal error occurred while processing the request",
    )

    # Force cleanup of any temporary data
    gc.collect()

    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode="json"),
    )
