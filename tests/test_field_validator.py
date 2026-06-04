"""Unit tests for app.validators.field_validator module."""

import pytest

from app.models.document import DocumentType
from app.models.validation import ValidationResult
from app.validators.field_validator import FieldValidator


@pytest.fixture
def validator() -> FieldValidator:
    """Create a FieldValidator instance for testing."""
    return FieldValidator()


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        result = ValidationResult(is_valid=True, normalized_value="01/01/2000")
        assert result.is_valid is True
        assert result.normalized_value == "01/01/2000"
        assert result.error_message is None

    def test_invalid_result(self) -> None:
        result = ValidationResult(is_valid=False, error_message="Bad format")
        assert result.is_valid is False
        assert result.normalized_value is None
        assert result.error_message == "Bad format"

    def test_defaults(self) -> None:
        result = ValidationResult(is_valid=True)
        assert result.normalized_value is None
        assert result.error_message is None


class TestValidateDate:
    """Tests for date normalization and validation."""

    def test_dd_slash_mm_slash_yyyy(self, validator: FieldValidator) -> None:
        result = validator.validate_date("15/06/1990")
        assert result.is_valid is True
        assert result.normalized_value == "15/06/1990"

    def test_dd_dash_mm_dash_yyyy(self, validator: FieldValidator) -> None:
        result = validator.validate_date("15-06-1990")
        assert result.is_valid is True
        assert result.normalized_value == "15/06/1990"

    def test_dd_dot_mm_dot_yyyy(self, validator: FieldValidator) -> None:
        result = validator.validate_date("15.06.1990")
        assert result.is_valid is True
        assert result.normalized_value == "15/06/1990"

    def test_yyyy_dash_mm_dash_dd_iso(self, validator: FieldValidator) -> None:
        result = validator.validate_date("1990-06-15")
        assert result.is_valid is True
        assert result.normalized_value == "15/06/1990"

    def test_yyyy_slash_mm_slash_dd(self, validator: FieldValidator) -> None:
        result = validator.validate_date("1990/06/15")
        assert result.is_valid is True
        assert result.normalized_value == "15/06/1990"

    def test_single_digit_day_and_month(self, validator: FieldValidator) -> None:
        result = validator.validate_date("5/3/2000")
        assert result.is_valid is True
        assert result.normalized_value == "05/03/2000"

    def test_zero_pads_output(self, validator: FieldValidator) -> None:
        result = validator.validate_date("1-1-2020")
        assert result.is_valid is True
        assert result.normalized_value == "01/01/2020"

    def test_invalid_month_13(self, validator: FieldValidator) -> None:
        result = validator.validate_date("15/13/1990")
        assert result.is_valid is False
        assert "Month" in result.error_message

    def test_invalid_day_0(self, validator: FieldValidator) -> None:
        result = validator.validate_date("0/06/1990")
        assert result.is_valid is False
        assert "Day" in result.error_message

    def test_invalid_day_32(self, validator: FieldValidator) -> None:
        result = validator.validate_date("32/06/1990")
        assert result.is_valid is False
        assert "Day" in result.error_message

    def test_invalid_year_below_1900(self, validator: FieldValidator) -> None:
        result = validator.validate_date("15/06/1800")
        assert result.is_valid is False
        assert "Year" in result.error_message

    def test_invalid_year_above_2100(self, validator: FieldValidator) -> None:
        result = validator.validate_date("15/06/2200")
        assert result.is_valid is False
        assert "Year" in result.error_message

    def test_unsupported_format(self, validator: FieldValidator) -> None:
        result = validator.validate_date("June 15, 1990")
        assert result.is_valid is False
        assert "does not match" in result.error_message

    def test_empty_string(self, validator: FieldValidator) -> None:
        result = validator.validate_date("")
        assert result.is_valid is False

    def test_whitespace_stripped(self, validator: FieldValidator) -> None:
        result = validator.validate_date("  15/06/1990  ")
        assert result.is_valid is True
        assert result.normalized_value == "15/06/1990"

    def test_boundary_year_1900(self, validator: FieldValidator) -> None:
        result = validator.validate_date("01/01/1900")
        assert result.is_valid is True

    def test_boundary_year_2100(self, validator: FieldValidator) -> None:
        result = validator.validate_date("31/12/2100")
        assert result.is_valid is True


class TestValidatePincode:
    """Tests for Indian pincode validation."""

    def test_valid_pincode(self, validator: FieldValidator) -> None:
        result = validator.validate_pincode("110001")
        assert result.is_valid is True
        assert result.normalized_value == "110001"

    def test_valid_pincode_starting_with_9(self, validator: FieldValidator) -> None:
        result = validator.validate_pincode("900001")
        assert result.is_valid is True
        assert result.normalized_value == "900001"

    def test_invalid_starts_with_zero(self, validator: FieldValidator) -> None:
        result = validator.validate_pincode("012345")
        assert result.is_valid is False
        assert "first digit" in result.error_message

    def test_invalid_too_short(self, validator: FieldValidator) -> None:
        result = validator.validate_pincode("12345")
        assert result.is_valid is False
        assert "6 digits" in result.error_message

    def test_invalid_too_long(self, validator: FieldValidator) -> None:
        result = validator.validate_pincode("1234567")
        assert result.is_valid is False
        assert "6 digits" in result.error_message

    def test_invalid_contains_letters(self, validator: FieldValidator) -> None:
        result = validator.validate_pincode("12345A")
        assert result.is_valid is False

    def test_empty_string(self, validator: FieldValidator) -> None:
        result = validator.validate_pincode("")
        assert result.is_valid is False

    def test_whitespace_stripped(self, validator: FieldValidator) -> None:
        result = validator.validate_pincode("  560001  ")
        assert result.is_valid is True
        assert result.normalized_value == "560001"


class TestValidateLicenseNumber:
    """Tests for Driving License number format validation."""

    def test_valid_karnataka_format(self, validator: FieldValidator) -> None:
        result = validator.validate_license_number("KA01 20170001234")
        assert result.is_valid is True
        assert result.normalized_value == "KA01 20170001234"

    def test_valid_delhi_format_with_dash(self, validator: FieldValidator) -> None:
        result = validator.validate_license_number("DL-0420110149646")
        assert result.is_valid is True

    def test_valid_maharashtra_format(self, validator: FieldValidator) -> None:
        result = validator.validate_license_number("MH12 20190001234")
        assert result.is_valid is True

    def test_valid_compact_format(self, validator: FieldValidator) -> None:
        result = validator.validate_license_number("TN0120190012345")
        assert result.is_valid is True

    def test_invalid_state_code(self, validator: FieldValidator) -> None:
        result = validator.validate_license_number("XX01 20170001234")
        assert result.is_valid is False
        assert "state code" in result.error_message.lower()

    def test_invalid_no_digits(self, validator: FieldValidator) -> None:
        result = validator.validate_license_number("KA")
        assert result.is_valid is False

    def test_invalid_lowercase_state(self, validator: FieldValidator) -> None:
        result = validator.validate_license_number("ka01 20170001234")
        assert result.is_valid is False

    def test_empty_string(self, validator: FieldValidator) -> None:
        result = validator.validate_license_number("")
        assert result.is_valid is False

    def test_whitespace_stripped(self, validator: FieldValidator) -> None:
        result = validator.validate_license_number("  KA01 20170001234  ")
        assert result.is_valid is True


class TestValidateAll:
    """Tests for validate_all orchestration."""

    def test_aadhaar_front_validates_dob(self, validator: FieldValidator) -> None:
        fields = {"dob": "15-06-1990", "name": "John Doe"}
        results = validator.validate_all(DocumentType.AADHAAR_FRONT, fields)
        assert "dob" in results
        assert results["dob"].is_valid is True
        assert results["dob"].normalized_value == "15/06/1990"
        # name should not be validated
        assert "name" not in results

    def test_aadhaar_back_validates_pincode(self, validator: FieldValidator) -> None:
        fields = {"pincode": "560001", "address": "123 Main St"}
        results = validator.validate_all(DocumentType.AADHAAR_BACK, fields)
        assert "pincode" in results
        assert results["pincode"].is_valid is True
        # address should not be validated
        assert "address" not in results

    def test_aadhaar_back_validates_dob_and_pincode(self, validator: FieldValidator) -> None:
        fields = {"dob": "1990-06-15", "pincode": "110001"}
        results = validator.validate_all(DocumentType.AADHAAR_BACK, fields)
        assert "dob" in results
        assert "pincode" in results
        assert results["dob"].normalized_value == "15/06/1990"

    def test_pan_card_validates_dob(self, validator: FieldValidator) -> None:
        fields = {"dob": "15.06.1990", "pan_number": "ABCDE1234F"}
        results = validator.validate_all(DocumentType.PAN_CARD, fields)
        assert "dob" in results
        assert results["dob"].is_valid is True
        # pan_number is not validated by validate_all (handled by extractor)
        assert "pan_number" not in results

    def test_driving_license_validates_all_dates_and_number(
        self, validator: FieldValidator
    ) -> None:
        fields = {
            "dob": "15/06/1990",
            "issue_date": "2020-01-10",
            "expiry_date": "2040-01-10",
            "license_number": "KA01 20170001234",
            "name": "John Doe",
        }
        results = validator.validate_all(DocumentType.DRIVING_LICENSE, fields)
        assert "dob" in results
        assert "issue_date" in results
        assert "expiry_date" in results
        assert "license_number" in results
        assert "name" not in results
        assert results["issue_date"].normalized_value == "10/01/2020"
        assert results["expiry_date"].normalized_value == "10/01/2040"

    def test_unknown_document_returns_empty(self, validator: FieldValidator) -> None:
        fields = {"dob": "15/06/1990", "pincode": "110001"}
        results = validator.validate_all(DocumentType.UNKNOWN, fields)
        assert results == {}

    def test_empty_fields_returns_empty(self, validator: FieldValidator) -> None:
        results = validator.validate_all(DocumentType.AADHAAR_FRONT, {})
        assert results == {}

    def test_none_field_values_skipped(self, validator: FieldValidator) -> None:
        fields = {"dob": None, "pincode": None}
        results = validator.validate_all(DocumentType.AADHAAR_BACK, fields)
        assert results == {}

    def test_invalid_fields_return_invalid_results(
        self, validator: FieldValidator
    ) -> None:
        fields = {"dob": "invalid-date", "pincode": "000000"}
        results = validator.validate_all(DocumentType.AADHAAR_BACK, fields)
        assert "dob" in results
        assert results["dob"].is_valid is False
        assert "pincode" in results
        assert results["pincode"].is_valid is False
