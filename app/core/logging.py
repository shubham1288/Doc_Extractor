"""
Structured logging configuration using structlog.

Provides JSON-formatted log output for production and console output
for development. Integrates with Python's standard library logging
so that third-party libraries also emit structured logs.

Environment Variables:
    LOG_LEVEL: Logging level (default: INFO)
    LOG_FORMAT: "json" for production, "console" for development (default: json)
"""

import logging
import os
import sys

import structlog

from app.middleware.pii_filter import redact_pii


def setup_logging() -> None:
    """
    Initialize structured logging configuration.

    Configures structlog with appropriate processors for the environment
    and integrates with Python's standard library logging. Call this once
    at application startup (e.g., in the FastAPI lifespan handler).
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", "json").lower()

    # Shared processors used by both structlog and stdlib integration
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        redact_pii,
    ]

    if log_format == "console":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to route through structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("paddle").setLevel(logging.WARNING)


def get_logger(name: str | None = None, **initial_bindings) -> structlog.stdlib.BoundLogger:
    """
    Return a bound structlog logger.

    Args:
        name: Optional logger name. Defaults to the calling module if not provided.
        **initial_bindings: Key-value pairs to bind to the logger permanently.

    Returns:
        A structlog BoundLogger instance with JSON output and the configured
        processor chain.

    Example:
        logger = get_logger("extraction")
        logger.info("document_processed", document_type="aadhaar", confidence=0.92)
    """
    logger = structlog.get_logger(name)
    if initial_bindings:
        logger = logger.bind(**initial_bindings)
    return logger
