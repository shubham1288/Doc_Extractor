"""
Property-based tests for API response structural completeness and request ID uniqueness.

Uses Hypothesis to verify that response models and request ID generation properties
hold for ALL generated inputs.

**Validates: Requirements 9.2, 9.6**
"""

from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.models.document import DocumentType
from app.schemas.responses import ErrorResponse, KYCExtractionResponse
from app.utils.request_id import generate_request_id


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: valid document types
document_type_strategy = st.sampled_from(list(DocumentType))

# Strategy: confidence scores between 0.0 and 1.0
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy: non-negative processing time in ms
processing_time_strategy = st.integers(min_value=0, max_value=100_000)

# Strategy: extracted data dict (arbitrary keys/values)
extracted_data_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "Nd", "Pc"))),
    values=st.one_of(st.text(max_size=100), st.integers(), st.none()),
    min_size=0,
    max_size=10,
)

# Strategy: request IDs starting with "req_"
request_id_strategy = st.builds(
    lambda suffix: f"req_{suffix}",
    st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "Nd"))),
)

# Strategy: non-empty strings for error fields
non_empty_string_strategy = st.text(min_size=1, max_size=100).filter(lambda s: s.strip())

# Strategy: datetime values
datetime_strategy = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2100, 1, 1),
    timezones=st.just(timezone.utc),
)

# Strategy: number of calls for uniqueness testing (2-1000)
num_calls_strategy = st.integers(min_value=2, max_value=1000)


# ---------------------------------------------------------------------------
# Property 17: API response structural completeness
# ---------------------------------------------------------------------------


class TestAPIResponseStructuralCompletenessProperty:
    """Property 17: API response structural completeness.

    For ANY valid KYCExtractionResponse, all required fields must be present
    and correctly typed. For ANY valid ErrorResponse, all required fields
    must be present with correct types.

    **Validates: Requirements 9.2**
    """

    @given(
        request_id=request_id_strategy,
        document_type=document_type_strategy,
        classification_confidence=confidence_strategy,
        extracted_data=extracted_data_strategy,
        processing_time_ms=processing_time_strategy,
        ocr_confidence=confidence_strategy,
        timestamp=datetime_strategy,
    )
    @settings(max_examples=500, deadline=2000)
    def test_kyc_response_has_all_required_fields(
        self,
        request_id: str,
        document_type: DocumentType,
        classification_confidence: float,
        extracted_data: dict,
        processing_time_ms: int,
        ocr_confidence: float,
        timestamp: datetime,
    ) -> None:
        """For any valid input data, KYCExtractionResponse contains all required fields."""
        response = KYCExtractionResponse(
            request_id=request_id,
            document_type=document_type,
            classification_confidence=classification_confidence,
            extracted_data=extracted_data,
            processing_time_ms=processing_time_ms,
            ocr_confidence=ocr_confidence,
            timestamp=timestamp,
        )

        # Verify all required fields are present and correctly typed
        assert isinstance(response.request_id, str)
        assert len(response.request_id) > 0
        assert response.request_id.startswith("req_")

        assert isinstance(response.document_type, DocumentType)
        assert response.document_type in DocumentType

        assert isinstance(response.classification_confidence, float)
        assert 0.0 <= response.classification_confidence <= 1.0

        assert isinstance(response.extracted_data, dict)

        assert isinstance(response.processing_time_ms, int)
        assert response.processing_time_ms >= 0

        assert isinstance(response.ocr_confidence, float)
        assert 0.0 <= response.ocr_confidence <= 1.0

        assert isinstance(response.timestamp, datetime)

    @given(
        request_id=request_id_strategy,
        document_type=document_type_strategy,
        classification_confidence=confidence_strategy,
        extracted_data=extracted_data_strategy,
        processing_time_ms=processing_time_strategy,
        ocr_confidence=confidence_strategy,
    )
    @settings(max_examples=300, deadline=2000)
    def test_kyc_response_request_id_starts_with_prefix(
        self,
        request_id: str,
        document_type: DocumentType,
        classification_confidence: float,
        extracted_data: dict,
        processing_time_ms: int,
        ocr_confidence: float,
    ) -> None:
        """For any KYCExtractionResponse, request_id starts with 'req_'."""
        response = KYCExtractionResponse(
            request_id=request_id,
            document_type=document_type,
            classification_confidence=classification_confidence,
            extracted_data=extracted_data,
            processing_time_ms=processing_time_ms,
            ocr_confidence=ocr_confidence,
        )

        assert response.request_id.startswith("req_"), (
            f"Expected request_id to start with 'req_', got '{response.request_id}'"
        )

    @given(
        request_id=request_id_strategy,
        document_type=document_type_strategy,
        classification_confidence=confidence_strategy,
        extracted_data=extracted_data_strategy,
        processing_time_ms=processing_time_strategy,
        ocr_confidence=confidence_strategy,
    )
    @settings(max_examples=300, deadline=2000)
    def test_kyc_response_confidence_scores_in_bounds(
        self,
        request_id: str,
        document_type: DocumentType,
        classification_confidence: float,
        extracted_data: dict,
        processing_time_ms: int,
        ocr_confidence: float,
    ) -> None:
        """For any KYCExtractionResponse, confidence scores are in [0.0, 1.0]."""
        response = KYCExtractionResponse(
            request_id=request_id,
            document_type=document_type,
            classification_confidence=classification_confidence,
            extracted_data=extracted_data,
            processing_time_ms=processing_time_ms,
            ocr_confidence=ocr_confidence,
        )

        assert 0.0 <= response.classification_confidence <= 1.0, (
            f"classification_confidence out of bounds: {response.classification_confidence}"
        )
        assert 0.0 <= response.ocr_confidence <= 1.0, (
            f"ocr_confidence out of bounds: {response.ocr_confidence}"
        )

    @given(
        request_id=request_id_strategy,
        error_code=non_empty_string_strategy,
        error_message=non_empty_string_strategy,
        timestamp=datetime_strategy,
    )
    @settings(max_examples=500, deadline=2000)
    def test_error_response_has_all_required_fields(
        self,
        request_id: str,
        error_code: str,
        error_message: str,
        timestamp: datetime,
    ) -> None:
        """For any valid ErrorResponse, all required fields must be present."""
        response = ErrorResponse(
            request_id=request_id,
            error_code=error_code,
            error_message=error_message,
            timestamp=timestamp,
        )

        # Verify all required fields are present and correctly typed
        assert isinstance(response.request_id, str)
        assert len(response.request_id) > 0
        assert response.request_id.startswith("req_")

        assert isinstance(response.error_code, str)
        assert len(response.error_code) > 0

        assert isinstance(response.error_message, str)
        assert len(response.error_message) > 0

        assert isinstance(response.timestamp, datetime)

    @given(
        request_id=request_id_strategy,
        error_code=non_empty_string_strategy,
        error_message=non_empty_string_strategy,
    )
    @settings(max_examples=300, deadline=2000)
    def test_error_response_request_id_starts_with_prefix(
        self,
        request_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """For any ErrorResponse, request_id starts with 'req_'."""
        response = ErrorResponse(
            request_id=request_id,
            error_code=error_code,
            error_message=error_message,
        )

        assert response.request_id.startswith("req_"), (
            f"Expected request_id to start with 'req_', got '{response.request_id}'"
        )


# ---------------------------------------------------------------------------
# Property 18: Request ID uniqueness
# ---------------------------------------------------------------------------


class TestRequestIDUniquenessProperty:
    """Property 18: Request ID uniqueness.

    For ANY number of calls (2-1000) to generate_request_id(), all results
    must be unique, start with "req_", and have a consistent format.

    **Validates: Requirements 9.6**
    """

    @given(num_calls=num_calls_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_all_request_ids_are_unique(self, num_calls: int) -> None:
        """For any number of calls (2-1000), all generated request IDs are unique."""
        ids = [generate_request_id() for _ in range(num_calls)]

        assert len(set(ids)) == len(ids), (
            f"Expected {num_calls} unique IDs but got {len(set(ids))} unique out of {len(ids)} total. "
            f"Duplicates found: {[x for x in ids if ids.count(x) > 1][:5]}"
        )

    @given(num_calls=num_calls_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_all_request_ids_start_with_prefix(self, num_calls: int) -> None:
        """For any number of calls, all request IDs start with 'req_'."""
        ids = [generate_request_id() for _ in range(num_calls)]

        for request_id in ids:
            assert request_id.startswith("req_"), (
                f"Expected request_id to start with 'req_', got '{request_id}'"
            )

    @given(num_calls=num_calls_strategy)
    @settings(max_examples=200, deadline=10000)
    def test_all_request_ids_have_consistent_format(self, num_calls: int) -> None:
        """For any number of calls, all request IDs have consistent length and character set."""
        ids = [generate_request_id() for _ in range(num_calls)]

        # All IDs should have the same length (req_ + 32 hex chars = 36)
        expected_length = 4 + 32  # "req_" prefix + uuid4 hex (32 chars)
        for request_id in ids:
            assert len(request_id) == expected_length, (
                f"Expected length {expected_length}, got {len(request_id)} for '{request_id}'"
            )

        # All IDs should consist of "req_" followed by only hexadecimal characters
        import re
        hex_pattern = re.compile(r"^req_[0-9a-f]{32}$")
        for request_id in ids:
            assert hex_pattern.match(request_id), (
                f"Request ID '{request_id}' does not match expected format 'req_<32 hex chars>'"
            )

    @given(num_calls=st.integers(min_value=2, max_value=50))
    @settings(max_examples=100, deadline=5000)
    def test_request_ids_are_non_empty(self, num_calls: int) -> None:
        """For any number of calls, all request IDs are non-empty strings."""
        ids = [generate_request_id() for _ in range(num_calls)]

        for request_id in ids:
            assert isinstance(request_id, str)
            assert len(request_id) > 0, "Request ID must not be empty"
