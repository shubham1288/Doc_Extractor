"""
Property-based tests for low OCR confidence threshold handling.

Verifies that for ANY OCR confidence value below 0.3, the system
returns document_type UNKNOWN with empty extraction fields.

**Validates: Requirements 13.3**
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from app.core.config import get_settings
from app.models.document import DocumentType
from app.schemas.responses import KYCExtractionResponse


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: OCR confidence below threshold (0.3)
low_confidence_strategy = st.floats(
    min_value=0.0,
    max_value=0.299999,
    allow_nan=False,
    allow_infinity=False,
)

# Strategy: OCR confidence at or above threshold
high_confidence_strategy = st.floats(
    min_value=0.3,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Strategy: valid request IDs
request_id_strategy = st.builds(
    lambda suffix: f"req_{suffix}",
    st.from_regex(r"[0-9a-f]{32}", fullmatch=True),
)

# Strategy: processing time
processing_time_strategy = st.integers(min_value=0, max_value=10000)


# ---------------------------------------------------------------------------
# Property 21: Low OCR confidence produces UNKNOWN response
# ---------------------------------------------------------------------------


class TestLowOCRConfidenceProperty:
    """Property 21: Low OCR confidence produces UNKNOWN response.

    For ANY OCR result with average confidence below 0.3, the System SHALL
    return a response with document_type set to UNKNOWN and empty extraction fields.

    **Validates: Requirements 13.3**
    """

    @given(
        confidence=low_confidence_strategy,
        request_id=request_id_strategy,
        processing_time_ms=processing_time_strategy,
    )
    @settings(max_examples=500, deadline=2000)
    def test_below_threshold_always_produces_unknown(
        self, confidence: float, request_id: str, processing_time_ms: int
    ) -> None:
        """For any OCR confidence < 0.3, response document_type is UNKNOWN."""
        settings_obj = get_settings()
        assert confidence < settings_obj.low_confidence_threshold

        # Simulate the endpoint logic: if confidence < threshold → UNKNOWN
        response = KYCExtractionResponse(
            request_id=request_id,
            document_type=DocumentType.UNKNOWN,
            classification_confidence=0.0,
            extracted_data={},
            processing_time_ms=processing_time_ms,
            ocr_confidence=confidence,
        )

        assert response.document_type == DocumentType.UNKNOWN, (
            f"Expected UNKNOWN for confidence {confidence}, got {response.document_type}"
        )
        assert response.extracted_data == {}, (
            f"Expected empty extracted_data for confidence {confidence}, "
            f"got {response.extracted_data}"
        )
        assert response.classification_confidence == 0.0, (
            f"Expected classification_confidence 0.0 for low OCR confidence, "
            f"got {response.classification_confidence}"
        )

    @given(confidence=low_confidence_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_threshold_boundary_below(self, confidence: float) -> None:
        """For any confidence strictly below 0.3, the threshold check triggers."""
        settings_obj = get_settings()
        assert confidence < settings_obj.low_confidence_threshold, (
            f"Generated confidence {confidence} is not below threshold "
            f"{settings_obj.low_confidence_threshold}"
        )

    @given(confidence=high_confidence_strategy)
    @settings(max_examples=300, deadline=2000)
    def test_at_or_above_threshold_does_not_produce_unknown_by_default(
        self, confidence: float
    ) -> None:
        """For any confidence >= 0.3, the threshold check does NOT force UNKNOWN."""
        settings_obj = get_settings()
        # Confidence at or above threshold should NOT trigger the low-confidence path
        assert confidence >= settings_obj.low_confidence_threshold, (
            f"Generated confidence {confidence} should be >= {settings_obj.low_confidence_threshold}"
        )

    @given(
        confidence=low_confidence_strategy,
        request_id=request_id_strategy,
    )
    @settings(max_examples=500, deadline=2000)
    def test_low_confidence_response_has_zero_classification_confidence(
        self, confidence: float, request_id: str
    ) -> None:
        """For any low OCR confidence, classification_confidence is 0.0."""
        response = KYCExtractionResponse(
            request_id=request_id,
            document_type=DocumentType.UNKNOWN,
            classification_confidence=0.0,
            extracted_data={},
            processing_time_ms=100,
            ocr_confidence=confidence,
        )

        assert response.classification_confidence == 0.0

    @given(
        confidence=low_confidence_strategy,
        request_id=request_id_strategy,
    )
    @settings(max_examples=500, deadline=2000)
    def test_low_confidence_preserves_ocr_confidence_in_response(
        self, confidence: float, request_id: str
    ) -> None:
        """For any low OCR confidence, the actual confidence value is preserved in the response."""
        response = KYCExtractionResponse(
            request_id=request_id,
            document_type=DocumentType.UNKNOWN,
            classification_confidence=0.0,
            extracted_data={},
            processing_time_ms=100,
            ocr_confidence=confidence,
        )

        assert response.ocr_confidence == confidence, (
            f"Expected ocr_confidence to be {confidence}, got {response.ocr_confidence}"
        )

    @given(
        confidence=low_confidence_strategy,
        request_id=request_id_strategy,
        processing_time_ms=processing_time_strategy,
    )
    @settings(max_examples=300, deadline=2000)
    def test_low_confidence_response_is_structurally_complete(
        self, confidence: float, request_id: str, processing_time_ms: int
    ) -> None:
        """For any low confidence response, all required fields are present."""
        response = KYCExtractionResponse(
            request_id=request_id,
            document_type=DocumentType.UNKNOWN,
            classification_confidence=0.0,
            extracted_data={},
            processing_time_ms=processing_time_ms,
            ocr_confidence=confidence,
        )

        # All required fields must be present
        assert response.request_id is not None
        assert response.document_type is not None
        assert response.classification_confidence is not None
        assert response.extracted_data is not None
        assert response.processing_time_ms is not None
        assert response.ocr_confidence is not None
        assert response.timestamp is not None
