"""
Property-based tests for Document Classifier — confidence bounds and threshold.

Uses Hypothesis to verify that classifier properties hold for ALL generated inputs.

**Validates: Requirements 4.2, 4.3**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.models.document import ClassificationResult, DocumentType
from app.models.ocr import OCRResult
from app.services.classifier import DocumentClassifier


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Keywords associated with each document type for generating realistic text
AADHAAR_FRONT_KEYWORDS = [
    "uidai", "unique identification", "government of india",
    "male", "female", "transgender", "dob", "date of birth",
    "year of birth", "1234 5678 9012",
]

AADHAAR_BACK_KEYWORDS = [
    "uidai", "address", "s/o", "d/o", "w/o", "house", "street",
    "nagar", "colony", "district", "village", "pincode", "560001",
    "1234 5678 9012",
]

PAN_KEYWORDS = [
    "income tax", "permanent account", "father", "government of india",
    "abcde1234f",
]

DRIVING_LICENSE_KEYWORDS = [
    "driving", "licence", "license", "transport", "rto",
    "regional transport", "validity", "expiry", "issue date",
    "motor vehicle", "vehicle class",
]

ALL_KEYWORDS = (
    AADHAAR_FRONT_KEYWORDS
    + AADHAAR_BACK_KEYWORDS
    + PAN_KEYWORDS
    + DRIVING_LICENSE_KEYWORDS
)


def _make_ocr_result(text: str) -> OCRResult:
    """Helper to create an OCRResult from text for classification."""
    return OCRResult(
        text=text,
        boxes=[],
        confidences=[0.9],
        avg_confidence=0.9,
    )


# Strategy: random text strings of various lengths including empty
random_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=500,
)

# Strategy: text built from random combinations of keywords
keyword_text_strategy = st.lists(
    st.sampled_from(ALL_KEYWORDS),
    min_size=0,
    max_size=6,
).map(lambda keywords: " ".join(keywords))


# ---------------------------------------------------------------------------
# Property 11: Classification confidence bounds and threshold
# ---------------------------------------------------------------------------


class TestConfidenceBoundsAndThresholdProperty:
    """Property 11: Classification confidence bounds and threshold.

    For ANY text input to classify():
    - The confidence score must be between 0.0 and 1.0 (inclusive)
    - If confidence < 0.5, document_type must be UNKNOWN
    - If document_type != UNKNOWN, confidence must be >= 0.5

    **Validates: Requirements 4.2, 4.3**
    """

    @given(text=random_text_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_confidence_within_bounds_for_random_text(self, text: str) -> None:
        """For any random text, confidence is in [0.0, 1.0]."""
        classifier = DocumentClassifier()
        ocr_result = _make_ocr_result(text)

        result = classifier.classify(ocr_result)

        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence {result.confidence} is outside [0.0, 1.0] "
            f"for text: {text!r}"
        )

    @given(text=random_text_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_low_confidence_implies_unknown_for_random_text(self, text: str) -> None:
        """For any random text, if confidence < 0.5 then document_type is UNKNOWN."""
        classifier = DocumentClassifier()
        ocr_result = _make_ocr_result(text)

        result = classifier.classify(ocr_result)

        if result.confidence < 0.5:
            assert result.document_type == DocumentType.UNKNOWN, (
                f"Confidence {result.confidence} < 0.5 but document_type is "
                f"{result.document_type} (expected UNKNOWN) for text: {text!r}"
            )

    @given(text=random_text_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_non_unknown_implies_sufficient_confidence_for_random_text(
        self, text: str
    ) -> None:
        """For any random text, if document_type != UNKNOWN then confidence >= 0.5."""
        classifier = DocumentClassifier()
        ocr_result = _make_ocr_result(text)

        result = classifier.classify(ocr_result)

        if result.document_type != DocumentType.UNKNOWN:
            assert result.confidence >= 0.5, (
                f"document_type is {result.document_type} (not UNKNOWN) but "
                f"confidence {result.confidence} < 0.5 for text: {text!r}"
            )

    @given(text=keyword_text_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_confidence_within_bounds_for_keyword_text(self, text: str) -> None:
        """For any keyword combination text, confidence is in [0.0, 1.0]."""
        classifier = DocumentClassifier()
        ocr_result = _make_ocr_result(text)

        result = classifier.classify(ocr_result)

        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence {result.confidence} is outside [0.0, 1.0] "
            f"for keyword text: {text!r}"
        )

    @given(text=keyword_text_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_low_confidence_implies_unknown_for_keyword_text(self, text: str) -> None:
        """For any keyword combination, if confidence < 0.5 then document_type is UNKNOWN."""
        classifier = DocumentClassifier()
        ocr_result = _make_ocr_result(text)

        result = classifier.classify(ocr_result)

        if result.confidence < 0.5:
            assert result.document_type == DocumentType.UNKNOWN, (
                f"Confidence {result.confidence} < 0.5 but document_type is "
                f"{result.document_type} (expected UNKNOWN) for keyword text: {text!r}"
            )

    @given(text=keyword_text_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_non_unknown_implies_sufficient_confidence_for_keyword_text(
        self, text: str
    ) -> None:
        """For any keyword combination, if document_type != UNKNOWN then confidence >= 0.5."""
        classifier = DocumentClassifier()
        ocr_result = _make_ocr_result(text)

        result = classifier.classify(ocr_result)

        if result.document_type != DocumentType.UNKNOWN:
            assert result.confidence >= 0.5, (
                f"document_type is {result.document_type} (not UNKNOWN) but "
                f"confidence {result.confidence} < 0.5 for keyword text: {text!r}"
            )

    @given(
        text=st.text(min_size=0, max_size=0)
    )
    @settings(max_examples=10, deadline=2000)
    def test_empty_text_returns_valid_result(self, text: str) -> None:
        """Empty text input still produces valid confidence bounds and threshold behavior."""
        classifier = DocumentClassifier()
        ocr_result = _make_ocr_result(text)

        result = classifier.classify(ocr_result)

        assert 0.0 <= result.confidence <= 1.0
        if result.confidence < 0.5:
            assert result.document_type == DocumentType.UNKNOWN
        if result.document_type != DocumentType.UNKNOWN:
            assert result.confidence >= 0.5
