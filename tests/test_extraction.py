"""Unit tests for field extraction engines.

Covers:
- Verhoeff checksum validation (known valid/invalid numbers)
- PAN format extraction
- BaseExtractor normalize_name
- Extraction router returns correct extractor for each type
- AadhaarExtractor basic extraction
- DrivingLicenseExtractor basic extraction
"""

import pytest

from app.extraction.aadhaar import AadhaarExtractor
from app.extraction.base import BaseExtractor, ExtractionField
from app.extraction.driving_license import DrivingLicenseExtractor
from app.extraction.pan import PANExtractor
from app.extraction.router import get_extractor
from app.extraction.verhoeff import validate_verhoeff
from app.models.document import DocumentType
from app.models.ocr import OCRResult


# ---------------------------------------------------------------------------
# Verhoeff Checksum Tests
# ---------------------------------------------------------------------------


class TestVerhoeffChecksum:
    """Tests for Verhoeff checksum validation algorithm."""

    def test_valid_aadhaar_number(self):
        """Known valid Aadhaar number passes Verhoeff check."""
        # 499118665246 is a known valid Verhoeff number
        assert validate_verhoeff("499118665246") is True

    def test_invalid_aadhaar_number_single_digit_change(self):
        """Changing a single digit in a valid number makes it invalid."""
        # Change last digit of valid number
        assert validate_verhoeff("499118665247") is False

    def test_invalid_aadhaar_number_transposition(self):
        """Swapping adjacent digits in a valid number makes it invalid."""
        # Swap first two digits of valid number (49 -> 94)
        assert validate_verhoeff("949118665246") is False

    def test_all_zeros_is_invalid(self):
        """000000000000 is not valid per Verhoeff algorithm."""
        assert validate_verhoeff("000000000000") is False

    def test_raises_on_non_digit_input(self):
        """Non-digit input raises ValueError."""
        with pytest.raises(ValueError):
            validate_verhoeff("12345678901a")

    def test_raises_on_wrong_length(self):
        """Input not exactly 12 digits raises ValueError."""
        with pytest.raises(ValueError):
            validate_verhoeff("1234567890")

        with pytest.raises(ValueError):
            validate_verhoeff("1234567890123")

    def test_raises_on_empty_string(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            validate_verhoeff("")

    def test_known_valid_numbers(self):
        """Multiple known valid Verhoeff numbers validate correctly."""
        valid_numbers = [
            "499118665246",
        ]
        for number in valid_numbers:
            assert validate_verhoeff(number) is True, f"{number} should be valid"

    def test_deterministic(self):
        """Same input always produces same output."""
        for _ in range(10):
            assert validate_verhoeff("499118665246") is True
            assert validate_verhoeff("499118665247") is False


# ---------------------------------------------------------------------------
# PAN Format Extraction Tests
# ---------------------------------------------------------------------------


class TestPANExtraction:
    """Tests for PAN number extraction."""

    def _make_ocr_result(self, text: str, confidence: float = 0.9) -> OCRResult:
        return OCRResult(text=text, boxes=[], confidences=[], avg_confidence=confidence)

    def test_valid_pan_extraction(self):
        """Extracts a valid PAN number from text."""
        text = "INCOME TAX DEPARTMENT\nPermanent Account Number\nABCDE1234F\nName: RAJESH KUMAR"
        ocr = self._make_ocr_result(text)
        extractor = PANExtractor()
        result = extractor.extract(ocr)

        assert result["pan_number"].value == "ABCDE1234F"
        assert result["pan_number"].confidence > 0.0

    def test_pan_with_surrounding_text(self):
        """PAN is extracted even with surrounding text on the same line."""
        text = "PAN: BGHPS4231K issued by IT Department"
        ocr = self._make_ocr_result(text)
        extractor = PANExtractor()
        result = extractor.extract(ocr)

        assert result["pan_number"].value == "BGHPS4231K"

    def test_no_pan_in_text(self):
        """Returns None when no PAN pattern is found."""
        text = "This text has no PAN number in it at all"
        ocr = self._make_ocr_result(text)
        extractor = PANExtractor()
        result = extractor.extract(ocr)

        assert result["pan_number"].value is None
        assert result["pan_number"].confidence == 0.0

    def test_pan_name_extraction(self):
        """Extracts name from PAN card text."""
        text = "INCOME TAX DEPARTMENT\nName: RAJESH KUMAR\nFather's Name: MOHAN KUMAR\nDOB: 15/08/1990\nABCDE1234F"
        ocr = self._make_ocr_result(text)
        extractor = PANExtractor()
        result = extractor.extract(ocr)

        assert result["name"].value == "Rajesh Kumar"

    def test_pan_father_name_extraction(self):
        """Extracts father's name from PAN card text."""
        text = "Name: RAJESH KUMAR\nFather's Name: MOHAN KUMAR\nDOB: 15/08/1990"
        ocr = self._make_ocr_result(text)
        extractor = PANExtractor()
        result = extractor.extract(ocr)

        assert result["father_name"].value == "Mohan Kumar"

    def test_pan_dob_extraction(self):
        """Extracts DOB from PAN card text."""
        text = "Name: RAJESH\nDOB: 15/08/1990\nABCDE1234F"
        ocr = self._make_ocr_result(text)
        extractor = PANExtractor()
        result = extractor.extract(ocr)

        assert result["dob"].value == "15/08/1990"


# ---------------------------------------------------------------------------
# BaseExtractor normalize_name Tests
# ---------------------------------------------------------------------------


class TestNormalizeName:
    """Tests for BaseExtractor._normalize_name helper."""

    def _get_extractor(self) -> BaseExtractor:
        """Get a concrete extractor to test base class methods."""
        return PANExtractor()

    def test_simple_name_title_case(self):
        """Converts all-caps name to title case."""
        extractor = self._get_extractor()
        assert extractor._normalize_name("RAJESH KUMAR") == "Rajesh Kumar"

    def test_preserves_single_initial(self):
        """Preserves single-letter initials as uppercase."""
        extractor = self._get_extractor()
        assert extractor._normalize_name("S KUMAR") == "S Kumar"

    def test_preserves_initial_with_dot(self):
        """Preserves initials with dots as uppercase."""
        extractor = self._get_extractor()
        assert extractor._normalize_name("S. KUMAR") == "S. Kumar"

    def test_collapses_whitespace(self):
        """Collapses multiple spaces to single space."""
        extractor = self._get_extractor()
        assert extractor._normalize_name("RAJESH   KUMAR") == "Rajesh Kumar"

    def test_strips_leading_trailing_whitespace(self):
        """Strips leading and trailing whitespace."""
        extractor = self._get_extractor()
        assert extractor._normalize_name("  RAJESH KUMAR  ") == "Rajesh Kumar"

    def test_removes_ocr_artifacts(self):
        """Removes common OCR artifacts (pipes, brackets, etc.)."""
        extractor = self._get_extractor()
        assert extractor._normalize_name("RAJESH |KUMAR|") == "Rajesh Kumar"

    def test_empty_string(self):
        """Returns empty string for empty input."""
        extractor = self._get_extractor()
        assert extractor._normalize_name("") == ""

    def test_mixed_case_input(self):
        """Handles mixed case input correctly."""
        extractor = self._get_extractor()
        assert extractor._normalize_name("rajesh Kumar MOHAN") == "Rajesh Kumar Mohan"


# ---------------------------------------------------------------------------
# Extraction Router Tests
# ---------------------------------------------------------------------------


class TestExtractionRouter:
    """Tests for extraction router mapping."""

    def test_aadhaar_front_returns_aadhaar_extractor(self):
        """AADHAAR_FRONT maps to AadhaarExtractor."""
        extractor = get_extractor(DocumentType.AADHAAR_FRONT)
        assert isinstance(extractor, AadhaarExtractor)

    def test_aadhaar_back_returns_aadhaar_extractor(self):
        """AADHAAR_BACK maps to AadhaarExtractor."""
        extractor = get_extractor(DocumentType.AADHAAR_BACK)
        assert isinstance(extractor, AadhaarExtractor)

    def test_pan_card_returns_pan_extractor(self):
        """PAN_CARD maps to PANExtractor."""
        extractor = get_extractor(DocumentType.PAN_CARD)
        assert isinstance(extractor, PANExtractor)

    def test_driving_license_returns_dl_extractor(self):
        """DRIVING_LICENSE maps to DrivingLicenseExtractor."""
        extractor = get_extractor(DocumentType.DRIVING_LICENSE)
        assert isinstance(extractor, DrivingLicenseExtractor)

    def test_unknown_raises_value_error(self):
        """UNKNOWN document type raises ValueError."""
        with pytest.raises(ValueError, match="No extractor available"):
            get_extractor(DocumentType.UNKNOWN)

    def test_router_returns_new_instances(self):
        """Each call returns a new extractor instance."""
        ext1 = get_extractor(DocumentType.PAN_CARD)
        ext2 = get_extractor(DocumentType.PAN_CARD)
        assert ext1 is not ext2


# ---------------------------------------------------------------------------
# AadhaarExtractor Tests
# ---------------------------------------------------------------------------


class TestAadhaarExtractor:
    """Tests for Aadhaar card field extraction."""

    def _make_ocr_result(self, text: str, confidence: float = 0.9) -> OCRResult:
        return OCRResult(text=text, boxes=[], confidences=[], avg_confidence=confidence)

    def test_extract_masked_aadhaar_number(self):
        """Extracts masked Aadhaar number (XXXX XXXX 1234)."""
        text = "Government of India\nXXXX XXXX 4567\nName: RAJESH"
        ocr = self._make_ocr_result(text)
        extractor = AadhaarExtractor()
        result = extractor.extract(ocr)

        assert result["aadhaar_number"].value == "XXXX XXXX 4567"
        assert result["aadhaar_number"].confidence > 0.0

    def test_extract_full_aadhaar_number(self):
        """Extracts full 12-digit Aadhaar number."""
        text = "Government of India\n4991 1866 5246\nName: RAJESH"
        ocr = self._make_ocr_result(text)
        extractor = AadhaarExtractor()
        result = extractor.extract(ocr)

        assert result["aadhaar_number"].value == "4991 1866 5246"
        assert result["aadhaar_number"].confidence > 0.0

    def test_extract_gender_male(self):
        """Extracts Male gender."""
        text = "DOB: 15/08/1990\nMale\n1234 5678 9012"
        ocr = self._make_ocr_result(text)
        extractor = AadhaarExtractor()
        result = extractor.extract(ocr)

        assert result["gender"].value == "Male"

    def test_extract_gender_female(self):
        """Extracts Female gender."""
        text = "DOB: 15/08/1990\nFemale\n1234 5678 9012"
        ocr = self._make_ocr_result(text)
        extractor = AadhaarExtractor()
        result = extractor.extract(ocr)

        assert result["gender"].value == "Female"

    def test_extract_dob(self):
        """Extracts DOB after keyword."""
        text = "Date of Birth: 15/08/1990\nMale"
        ocr = self._make_ocr_result(text)
        extractor = AadhaarExtractor()
        result = extractor.extract(ocr)

        assert result["dob"].value == "15/08/1990"

    def test_extract_pincode(self):
        """Extracts 6-digit pincode."""
        text = "Address: 123 Main Road\nMumbai\nPin Code: 400001"
        ocr = self._make_ocr_result(text)
        extractor = AadhaarExtractor()
        result = extractor.extract(ocr)

        assert result["pincode"].value == "400001"

    def test_extract_state(self):
        """Extracts state name from address text."""
        text = "Address: 123 Main Road\nMumbai, Maharashtra - 400001"
        ocr = self._make_ocr_result(text)
        extractor = AadhaarExtractor()
        result = extractor.extract(ocr)

        assert result["state"].value == "Maharashtra"

    def test_confidence_within_bounds(self):
        """All confidence scores are between 0.0 and 1.0."""
        text = "Government of India\nXXXX XXXX 4567\nName: RAJESH\nDOB: 15/08/1990\nMale"
        ocr = self._make_ocr_result(text, confidence=0.9)
        extractor = AadhaarExtractor()
        result = extractor.extract(ocr)

        for field_name, field in result.items():
            assert 0.0 <= field.confidence <= 1.0, (
                f"Field '{field_name}' confidence {field.confidence} out of bounds"
            )


# ---------------------------------------------------------------------------
# DrivingLicenseExtractor Tests
# ---------------------------------------------------------------------------


class TestDrivingLicenseExtractor:
    """Tests for Driving License field extraction."""

    def _make_ocr_result(self, text: str, confidence: float = 0.9) -> OCRResult:
        return OCRResult(text=text, boxes=[], confidences=[], avg_confidence=confidence)

    def test_extract_dl_number(self):
        """Extracts DL number in standard format."""
        text = "Driving License\nDL No: MH0220170012345\nName: RAJESH KUMAR"
        ocr = self._make_ocr_result(text)
        extractor = DrivingLicenseExtractor()
        result = extractor.extract(ocr)

        assert result["license_number"].value is not None
        assert result["license_number"].confidence > 0.0

    def test_extract_issue_date(self):
        """Extracts issue date from DL text."""
        text = "Issue Date: 01/05/2017\nExpiry Date: 01/05/2037"
        ocr = self._make_ocr_result(text)
        extractor = DrivingLicenseExtractor()
        result = extractor.extract(ocr)

        assert result["issue_date"].value == "01/05/2017"

    def test_extract_expiry_date(self):
        """Extracts expiry date from DL text."""
        text = "Issue Date: 01/05/2017\nValid Till: 01/05/2037"
        ocr = self._make_ocr_result(text)
        extractor = DrivingLicenseExtractor()
        result = extractor.extract(ocr)

        assert result["expiry_date"].value == "01/05/2037"

    def test_extract_issuing_authority(self):
        """Extracts issuing authority (RTO)."""
        text = "Issuing Authority: RTO Mumbai Central"
        ocr = self._make_ocr_result(text)
        extractor = DrivingLicenseExtractor()
        result = extractor.extract(ocr)

        assert result["issuing_authority"].value == "RTO Mumbai Central"

    def test_confidence_within_bounds(self):
        """All confidence scores are between 0.0 and 1.0."""
        text = "DL No: KA0120190012345\nName: RAJESH\nDOB: 15/08/1990\nIssue Date: 01/01/2019\nExpiry Date: 01/01/2039"
        ocr = self._make_ocr_result(text, confidence=0.85)
        extractor = DrivingLicenseExtractor()
        result = extractor.extract(ocr)

        for field_name, field in result.items():
            assert 0.0 <= field.confidence <= 1.0, (
                f"Field '{field_name}' confidence {field.confidence} out of bounds"
            )


# ---------------------------------------------------------------------------
# BaseExtractor fuzzy_match Tests
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    """Tests for BaseExtractor._fuzzy_match helper."""

    def _get_extractor(self) -> BaseExtractor:
        return PANExtractor()

    def test_exact_match(self):
        """Exact substring match is found."""
        extractor = self._get_extractor()
        result = extractor._fuzzy_match("Hello World", "World")
        assert result == "World"

    def test_fuzzy_match_with_ocr_noise(self):
        """Fuzzy match handles minor OCR corruption."""
        extractor = self._get_extractor()
        # "Werld" is similar to "World" (1 char diff in 5 = 0.8 ratio)
        result = extractor._fuzzy_match("Hello Werld", "World", threshold=0.7)
        assert result is not None

    def test_no_match_below_threshold(self):
        """Returns None when best match is below threshold."""
        extractor = self._get_extractor()
        result = extractor._fuzzy_match("completely different", "World", threshold=0.8)
        assert result is None

    def test_empty_text(self):
        """Returns None for empty text."""
        extractor = self._get_extractor()
        assert extractor._fuzzy_match("", "World") is None

    def test_empty_pattern(self):
        """Returns None for empty pattern."""
        extractor = self._get_extractor()
        assert extractor._fuzzy_match("Hello World", "") is None
