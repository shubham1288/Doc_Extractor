"""
Request logging middleware for FastAPI.

Provides structured logging of HTTP requests with:
- Unique request_id generation and attachment
- Request start logging (method, path)
- Request completion logging (status_code, processing_time_ms)
- No file contents or PII in log output

Usage:
    from app.middleware.request_logging import RequestLoggingMiddleware

    app.add_middleware(RequestLoggingMiddleware)
"""

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.utils.request_id import generate_request_id

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs request start/completion with timing and request_id.

    Attaches a unique request_id to each request via `request.state` and
    logs structured entries for request start and completion. Does NOT log
    any file contents or PII.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process the request with logging instrumentation.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware/endpoint to call.

        Returns:
            The HTTP response from downstream handlers.
        """
        request_id = generate_request_id()
        request.state.request_id = request_id

        start_time = time.perf_counter()

        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception:
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                processing_time_ms=processing_time_ms,
            )
            raise

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            processing_time_ms=processing_time_ms,
        )

        # Add request_id to response headers for traceability
        response.headers["X-Request-ID"] = request_id

        return response
