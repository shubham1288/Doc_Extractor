"""Unit tests for the Document Classifier module."""

import pytest

from app.models.document import ClassificationResult, DocumentType
from app.models.ocr import OCRResult
from app.services.classifier import DocumentClassifier, _normalize_text


# ---------------------------------------------------------------------------
# Text Normalization Tests (Task 5.1)
# ---------------------------------------------------------------------------


class TestNormalizeText:
    """Tests for _normalize_text helper function."""

    def test_converts_to_lowercase(self):
        assert _normalize_text("HELLO WORLD") == "hello world"

    def test_collapses_multiple_spaces(self):
        assert _normalize_text("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert _normalize_text("hello\t\n  world") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert _normalize_text("  hello world  ") == "hello world"

    def test_handles_empty_string(self):
        assert _normalize_text("") == ""

    def test_handles_whitespace_only(self):
        assert _normalize_text("   \t\n  ") == ""

    def test_mixed_case_and_whitespace(self):
        assert _normalize_text("  UIDAI   Government  OF  India  ") == "uidai government of india"


# ---------------------------------------------------------------------------
# Helper to build OCRResult
# ---------------------------------------------------------------------------


def _make_ocr_result(text: str) -> OCRResult:
    """Create a minimal OCRResult with given text."""
    return OCRResult(
        text=text,
        boxes=[],
        confidences=[],
        avg_confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Aadhaar Front Scoring Tests (Task 5.2)
# ---------------------------------------------------------------------------


class TestAadhaarFrontScoring:
    """Tests for Aadhaar front card classification."""

    def setup_method(self):
        self.classifier = DocumentClassifier()

    def test_full_aadhaar_front_text(self):
        text = """
        Government of India
        UIDAI Unique Identification Authority of India
        Name: Rahul Kumar
        DOB: 15/03/1990
        Male
        1234 5678 9012
        """
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.document_type == DocumentType.AADHAAR_FRONT
        assert result.confidence >= 0.5

    def test_aadhaar_number_pattern_detected(self):
        text = "UIDAI 1234 5678 9012 Male DOB: 01/01/1990"
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.document_type == DocumentType.AADHAAR_FRONT
        assert "aadhaar_number_pattern" in result.matched_patterns

    def test_uidai_keyword_detected(self):
        text = "UIDAI 1234 5678 9012 Male DOB: 01/01/1990"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "uidai_keyword" in result.matched_patterns

    def test_gender_keyword_detected(self):
        text = "UIDAI 1234 5678 9012 Female Date of Birth"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "gender_keyword" in result.matched_patterns

    def test_dob_keyword_detected(self):
        text = "UIDAI 1234 5678 9012 Male Date of Birth: 15/03/1990"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "dob_keyword" in result.matched_patterns


# ---------------------------------------------------------------------------
# Aadhaar Back Scoring Tests (Task 5.3)
# ---------------------------------------------------------------------------


class TestAadhaarBackScoring:
    """Tests for Aadhaar back card classification."""

    def setup_method(self):
        self.classifier = DocumentClassifier()

    def test_full_aadhaar_back_text(self):
        text = """
        UIDAI
        Address: 123 MG Road, Nagar Colony
        District XYZ
        State ABC
        Pincode 400001
        1234 5678 9012
        """
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.document_type == DocumentType.AADHAAR_BACK
        assert result.confidence >= 0.5

    def test_address_keyword_detected(self):
        text = "UIDAI Address: 123 Street 400001 1234 5678 9012"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "address_keyword" in result.matched_patterns

    def test_pincode_pattern_detected(self):
        text = "UIDAI Address: 123 Street 400001 1234 5678 9012"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "pincode_pattern" in result.matched_patterns

    def test_distinguishes_back_from_front(self):
        """Back should score higher when address is present but no gender/dob."""
        text = "UIDAI 1234 5678 9012 Address: S/O Rajesh, Colony Nagar 560001"
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.document_type == DocumentType.AADHAAR_BACK


# ---------------------------------------------------------------------------
# PAN Card Scoring Tests (Task 5.4)
# ---------------------------------------------------------------------------


class TestPANScoring:
    """Tests for PAN card classification."""

    def setup_method(self):
        self.classifier = DocumentClassifier()

    def test_full_pan_text(self):
        text = """
        Income Tax Department
        Government of India
        Permanent Account Number Card
        ABCDE1234F
        Name: Rahul Kumar
        Father's Name: Suresh Kumar
        Date of Birth: 15/03/1990
        """
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.document_type == DocumentType.PAN_CARD
        assert result.confidence >= 0.5

    def test_pan_number_pattern_detected(self):
        text = "Income Tax ABCDE1234F Father's Name"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "pan_number_pattern" in result.matched_patterns

    def test_income_tax_keyword_detected(self):
        text = "Income Tax Department ABCDE1234F Father"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "income_tax_keyword" in result.matched_patterns

    def test_father_keyword_detected(self):
        text = "Income Tax ABCDE1234F Father's Name: Suresh"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "father_keyword" in result.matched_patterns

    def test_govt_of_india_keyword_detected(self):
        text = "Government of India Income Tax ABCDE1234F Father"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "government_of_india" in result.matched_patterns


# ---------------------------------------------------------------------------
# Driving License Scoring Tests (Task 5.5)
# ---------------------------------------------------------------------------


class TestDrivingLicenseScoring:
    """Tests for Driving License classification."""

    def setup_method(self):
        self.classifier = DocumentClassifier()

    def test_full_dl_text(self):
        text = """
        Driving Licence
        Regional Transport Office
        DL Number: MH-02-2019-0012345
        Name: Rahul Kumar
        Issue Date: 01/01/2019
        Validity: 01/01/2039
        Class of Vehicle: LMV
        Motor Vehicle Act
        """
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.document_type == DocumentType.DRIVING_LICENSE
        assert result.confidence >= 0.5

    def test_driving_keyword_detected(self):
        text = "Driving Licence Transport Issue Date MH-02-20190012345"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "driving_licence_keyword" in result.matched_patterns

    def test_transport_keyword_detected(self):
        text = "Driving Licence RTO Motor Vehicle MH-02-20190012345"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "transport_keyword" in result.matched_patterns

    def test_validity_keyword_detected(self):
        text = "Driving Licence Transport Validity MH-02-20190012345"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "validity_keyword" in result.matched_patterns

    def test_motor_vehicle_keyword_detected(self):
        text = "Driving Licence Transport Motor Vehicle MH-02-20190012345"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "motor_vehicle_keyword" in result.matched_patterns

    def test_dl_number_pattern_detected(self):
        text = "Driving Licence Transport Motor Vehicle MH-02-20190012345"
        result = self.classifier.classify(_make_ocr_result(text))
        assert "dl_number_pattern" in result.matched_patterns


# ---------------------------------------------------------------------------
# Classify Method & Threshold Tests (Task 5.6)
# ---------------------------------------------------------------------------


class TestClassifyThreshold:
    """Tests for classify() method threshold logic."""

    def setup_method(self):
        self.classifier = DocumentClassifier()

    def test_unknown_when_no_patterns_match(self):
        text = "random text with nothing useful"
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.document_type == DocumentType.UNKNOWN
        assert result.confidence < 0.5

    def test_unknown_when_score_below_threshold(self):
        # Only one weak pattern match shouldn't trigger classification
        text = "some text with male keyword only"
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.document_type == DocumentType.UNKNOWN

    def test_confidence_capped_at_1(self):
        # Provide all patterns for a very high score
        text = """
        Government of India UIDAI Unique Identification
        1234 5678 9012 Male Date of Birth 01/01/1990
        Income Tax ABCDE1234F Father Name
        Driving Licence Transport Motor Vehicle
        """
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.confidence <= 1.0

    def test_returns_classification_result_type(self):
        text = "UIDAI 1234 5678 9012 Male DOB"
        result = self.classifier.classify(_make_ocr_result(text))
        assert isinstance(result, ClassificationResult)

    def test_matched_patterns_empty_for_unknown(self):
        text = "completely unrelated gibberish"
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.matched_patterns == []

    def test_highest_scoring_type_wins(self):
        """When multiple types match, the highest scoring should win."""
        # PAN-specific text should classify as PAN
        text = "Income Tax Permanent Account Number ABCDE1234F Father's Name Govt of India"
        result = self.classifier.classify(_make_ocr_result(text))
        assert result.document_type == DocumentType.PAN_CARD

    def test_empty_text_returns_unknown(self):
        result = self.classifier.classify(_make_ocr_result(""))
        assert result.document_type == DocumentType.UNKNOWN
        assert result.confidence == 0.0
