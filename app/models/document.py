"""Document classification data models."""

from dataclasses import dataclass, field
from enum import Enum


class DocumentType(str, Enum):
    """Supported Indian KYC document types."""

    AADHAAR_FRONT = "aadhaar_front"
    AADHAAR_BACK = "aadhaar_back"
    PAN_CARD = "pan_card"
    DRIVING_LICENSE = "driving_license"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of document classification.

    Attributes:
        document_type: The identified document type.
        confidence: Confidence score between 0.0 and 1.0.
        matched_patterns: List of pattern names that contributed to classification.
    """

    document_type: DocumentType
    confidence: float
    matched_patterns: list[str] = field(default_factory=list)
