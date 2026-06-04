"""
Pydantic v2 response models for the KYC Document Extraction API.

Defines structured response schemas for successful extraction results
and error responses. All responses include a unique request_id and timestamp.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.models.document import DocumentType


class KYCExtractionResponse(BaseModel):
    """Main API response model for successful document extraction.

    Attributes:
        request_id: Unique identifier for this request.
        document_type: The classified document type.
        classification_confidence: Confidence score of classification (0.0–1.0).
        extracted_data: Dictionary of extracted field names to values.
        processing_time_ms: Total processing time in milliseconds.
        ocr_confidence: Average OCR confidence score (0.0–1.0).
        timestamp: UTC timestamp of when the response was generated.
    """

    request_id: str = Field(description="Unique request identifier")
    document_type: DocumentType = Field(description="Classified document type")
    classification_confidence: float = Field(
        ge=0.0, le=1.0, description="Classification confidence score"
    )
    extracted_data: dict = Field(
        description="Extracted field data as key-value pairs"
    )
    processing_time_ms: int = Field(
        ge=0, description="Processing time in milliseconds"
    )
    ocr_confidence: float = Field(
        ge=0.0, le=1.0, description="Average OCR confidence score"
    )
    ocr_text: str | None = Field(
        default=None, description="Raw OCR text (included when debug=True in settings)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response generation timestamp (UTC)",
    )


class ErrorResponse(BaseModel):
    """Structured error response model.

    Never includes PII — only generic error information and the request_id
    for log correlation.

    Attributes:
        request_id: Unique identifier for this request.
        error_code: Machine-readable error code string.
        error_message: Human-readable error description (no PII).
        details: Optional additional error context (no PII).
        timestamp: UTC timestamp of when the error occurred.
    """

    request_id: str = Field(description="Unique request identifier")
    error_code: str = Field(description="Machine-readable error code")
    error_message: str = Field(description="Human-readable error description")
    details: dict | None = Field(
        default=None, description="Optional additional error context"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error occurrence timestamp (UTC)",
    )
