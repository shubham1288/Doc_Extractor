"""
Property-based tests for Field Validators — date normalization round-trip
and pincode validation.

Uses Hypothesis to verify that validation properties hold for ALL generated inputs.

**Validates: Requirements 8.1, 8.2**
"""

import re

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.validators.field_validator import FieldValidator


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: generate valid date components
valid_day = st.integers(min_value=1, max_value=31)
valid_month = st.integers(min_value=1, max_value=12)
valid_year = st.integers(min_value=1900, max_value=2100)

# Strategy: generate date format choice
date_format_choice = st.sampled_from(["DD/MM/YYYY", "DD-MM-YYYY", "DD.MM.YYYY", "YYYY-MM-DD", "YYYY/MM/DD"])

# Strategy: generate valid pincode (6 digits, first digit 1-9)
valid_pincode_strategy = st.from_regex(r"[1-9][0-9]{5}", fullmatch=True)

# Strategy: generate strings that are NOT valid pincodes
# Invalid pincodes: not 6 digits, or first digit is 0
invalid_pincode_strategy = st.one_of(
    # Too short (1-5 digits)
    st.from_regex(r"[0-9]{1,5}", fullmatch=True),
    # Too long (7+ digits)
    st.from_regex(r"[0-9]{7,12}", fullmatch=True),
    # 6 digits but first digit is 0
    st.from_regex(r"0[0-9]{5}", fullmatch=True),
    # Contains non-digit characters
    st.from_regex(r"[a-zA-Z][0-9]{5}", fullmatch=True),
    # Mixed alphanumeric
    st.from_regex(r"[0-9]{3}[a-zA-Z][0-9]{2}", fullmatch=True),
    # Empty string
    st.just(""),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_date(day: int, month: int, year: int, fmt: str) -> str:
    """Format a date according to the given format string."""
    if fmt == "DD/MM/YYYY":
        return f"{day:02d}/{month:02d}/{year}"
    elif fmt == "DD-MM-YYYY":
        return f"{day:02d}-{month:02d}-{year}"
    elif fmt == "DD.MM.YYYY":
        return f"{day:02d}.{month:02d}.{year}"
    elif fmt == "YYYY-MM-DD":
        return f"{year}-{month:02d}-{day:02d}"
    elif fmt == "YYYY/MM/DD":
        return f"{year}/{month:02d}/{day:02d}"
    else:
        raise ValueError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------------------
# Property 15: Date normalization round-trip
# ---------------------------------------------------------------------------


class TestDateNormalizationRoundTripProperty:
    """Property 15: Date normalization round-trip.

    For ANY valid date (day 1-31, month 1-12, year 1900-2100) formatted as
    DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD, or YYYY/MM/DD:
    - validate_date() must return is_valid=True
    - The normalized_value must always be in DD/MM/YYYY format
    - Normalizing the normalized_value again must produce the same result (idempotence)

    **Validates: Requirements 8.1**
    """

    @given(day=valid_day, month=valid_month, year=valid_year, fmt=date_format_choice)
    @settings(max_examples=500, deadline=2000)
    def test_valid_date_is_accepted(self, day: int, month: int, year: int, fmt: str) -> None:
        """For any valid date components in any supported format, validate_date returns is_valid=True."""
        date_str = _format_date(day, month, year, fmt)
        validator = FieldValidator()

        result = validator.validate_date(date_str)

        assert result.is_valid is True, (
            f"Expected is_valid=True for date '{date_str}' (day={day}, month={month}, "
            f"year={year}, format={fmt}) but got is_valid=False: {result.error_message}"
        )

    @given(day=valid_day, month=valid_month, year=valid_year, fmt=date_format_choice)
    @settings(max_examples=500, deadline=2000)
    def test_normalized_value_is_dd_mm_yyyy_format(self, day: int, month: int, year: int, fmt: str) -> None:
        """For any valid date, the normalized_value is always in DD/MM/YYYY format."""
        date_str = _format_date(day, month, year, fmt)
        validator = FieldValidator()

        result = validator.validate_date(date_str)

        assert result.is_valid is True
        # Verify the normalized_value matches DD/MM/YYYY pattern
        assert re.match(r"^\d{2}/\d{2}/\d{4}$", result.normalized_value), (
            f"Normalized value '{result.normalized_value}' does not match DD/MM/YYYY format "
            f"for input '{date_str}' (format={fmt})"
        )

    @given(day=valid_day, month=valid_month, year=valid_year, fmt=date_format_choice)
    @settings(max_examples=500, deadline=2000)
    def test_normalized_value_preserves_date_components(self, day: int, month: int, year: int, fmt: str) -> None:
        """For any valid date, the normalized_value contains the same day, month, year."""
        date_str = _format_date(day, month, year, fmt)
        validator = FieldValidator()

        result = validator.validate_date(date_str)

        assert result.is_valid is True
        # Parse the normalized value and check components
        parts = result.normalized_value.split("/")
        normalized_day = int(parts[0])
        normalized_month = int(parts[1])
        normalized_year = int(parts[2])

        assert normalized_day == day, (
            f"Day mismatch: input day={day}, normalized day={normalized_day} "
            f"for input '{date_str}'"
        )
        assert normalized_month == month, (
            f"Month mismatch: input month={month}, normalized month={normalized_month} "
            f"for input '{date_str}'"
        )
        assert normalized_year == year, (
            f"Year mismatch: input year={year}, normalized year={normalized_year} "
            f"for input '{date_str}'"
        )

    @given(day=valid_day, month=valid_month, year=valid_year, fmt=date_format_choice)
    @settings(max_examples=500, deadline=2000)
    def test_normalization_is_idempotent(self, day: int, month: int, year: int, fmt: str) -> None:
        """Normalizing the normalized_value again produces the same result (idempotence)."""
        date_str = _format_date(day, month, year, fmt)
        validator = FieldValidator()

        # First normalization
        result1 = validator.validate_date(date_str)
        assert result1.is_valid is True

        # Second normalization (normalize the normalized output)
        result2 = validator.validate_date(result1.normalized_value)

        assert result2.is_valid is True, (
            f"Normalizing the normalized value '{result1.normalized_value}' "
            f"returned is_valid=False: {result2.error_message}"
        )
        assert result2.normalized_value == result1.normalized_value, (
            f"Idempotence violated: first normalization produced '{result1.normalized_value}', "
            f"second normalization produced '{result2.normalized_value}' "
            f"for original input '{date_str}'"
        )


# ---------------------------------------------------------------------------
# Property 16: Pincode validation
# ---------------------------------------------------------------------------


class TestPincodeValidationProperty:
    """Property 16: Pincode validation.

    For ANY 6-digit string where first digit is 1-9:
    - validate_pincode() must return is_valid=True
    For ANY string not matching 6-digits-first-digit-1-9:
    - validate_pincode() must return is_valid=False

    **Validates: Requirements 8.2**
    """

    @given(pincode=valid_pincode_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_valid_pincode_is_accepted(self, pincode: str) -> None:
        """For any 6-digit string with first digit 1-9, validate_pincode returns is_valid=True."""
        validator = FieldValidator()

        result = validator.validate_pincode(pincode)

        assert result.is_valid is True, (
            f"Expected is_valid=True for valid pincode '{pincode}' "
            f"but got is_valid=False: {result.error_message}"
        )

    @given(pincode=valid_pincode_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_valid_pincode_normalized_value_equals_input(self, pincode: str) -> None:
        """For any valid pincode, the normalized_value equals the input."""
        validator = FieldValidator()

        result = validator.validate_pincode(pincode)

        assert result.is_valid is True
        assert result.normalized_value == pincode, (
            f"Expected normalized_value='{pincode}' "
            f"but got '{result.normalized_value}'"
        )

    @given(pincode=invalid_pincode_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_invalid_pincode_is_rejected(self, pincode: str) -> None:
        """For any string not matching valid pincode format, validate_pincode returns is_valid=False."""
        validator = FieldValidator()

        result = validator.validate_pincode(pincode)

        assert result.is_valid is False, (
            f"Expected is_valid=False for invalid pincode '{pincode}' "
            f"but got is_valid=True with normalized_value='{result.normalized_value}'"
        )

    @given(pincode=invalid_pincode_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_invalid_pincode_has_error_message(self, pincode: str) -> None:
        """For any invalid pincode, the result includes an error_message."""
        validator = FieldValidator()

        result = validator.validate_pincode(pincode)

        assert result.is_valid is False
        assert result.error_message is not None and len(result.error_message) > 0, (
            f"Expected non-empty error_message for invalid pincode '{pincode}' "
            f"but got: {result.error_message!r}"
        )

    @given(pincode=valid_pincode_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_valid_pincode_no_error_message(self, pincode: str) -> None:
        """For any valid pincode, error_message is None."""
        validator = FieldValidator()

        result = validator.validate_pincode(pincode)

        assert result.is_valid is True
        assert result.error_message is None, (
            f"Expected error_message=None for valid pincode '{pincode}' "
            f"but got: {result.error_message!r}"
        )
