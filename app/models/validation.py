"""Validation result data models."""

from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a field validation operation.

    Attributes:
        is_valid: Whether the field value passed validation.
        normalized_value: The normalized form of the value (if valid), or None.
        error_message: Description of the validation failure (if invalid), or None.
    """

    is_valid: bool
    normalized_value: str | None = None
    error_message: str | None = None
