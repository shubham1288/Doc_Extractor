"""
PII filter for structured logging.

Provides a structlog processor that redacts Aadhaar and PAN number
patterns from log entries before they reach the output renderer.

This ensures no PII (Personally Identifiable Information) is ever
written to log files, console, or log aggregation systems.

Usage:
    Add `redact_pii` to your structlog processor chain:

        structlog.configure(
            processors=[
                ...,
                redact_pii,
                ...,
            ],
        )
"""

import re
from typing import Any

# Pattern matching Aadhaar numbers: 4 digits, optional space, 4 digits, optional space, 4 digits
AADHAAR_PATTERN = re.compile(r"\d{4}\s?\d{4}\s?\d{4}")

# Pattern matching PAN numbers: 5 uppercase letters, 4 digits, 1 uppercase letter
PAN_PATTERN = re.compile(r"[A-Z]{5}\d{4}[A-Z]")

REDACTED_AADHAAR = "[REDACTED_AADHAAR]"
REDACTED_PAN = "[REDACTED_PAN]"


def _redact_value(value: Any) -> Any:
    """Redact PII patterns from a single value.

    Only processes string values. Non-string values are returned unchanged.
    """
    if not isinstance(value, str):
        return value

    # Redact PAN first (more specific pattern) to avoid partial matches
    result = PAN_PATTERN.sub(REDACTED_PAN, value)
    result = AADHAAR_PATTERN.sub(REDACTED_AADHAAR, result)
    return result


def redact_pii(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor that redacts Aadhaar and PAN patterns from all log values.

    Scans every value in the event dict (including the event message itself)
    and replaces any detected PII patterns with redaction placeholders.

    Args:
        logger: The wrapped logger object (unused, required by structlog API).
        method_name: The name of the log method called (unused).
        event_dict: The structured log event dictionary.

    Returns:
        The event dict with all PII patterns redacted.
    """
    for key in list(event_dict.keys()):
        event_dict[key] = _redact_value(event_dict[key])
    return event_dict
