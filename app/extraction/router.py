"""Extraction router: maps DocumentType to the appropriate extractor.

Provides a factory function that returns the correct extractor instance
based on the classified document type.
"""

from app.extraction.aadhaar import AadhaarExtractor
from app.extraction.base import BaseExtractor
from app.extraction.driving_license import DrivingLicenseExtractor
from app.extraction.pan import PANExtractor
from app.models.document import DocumentType


def get_extractor(document_type: DocumentType) -> BaseExtractor:
    """Get the appropriate extractor for a given document type.

    Maps each DocumentType to its specialized extractor class.
    Both Aadhaar front and back use the same AadhaarExtractor since
    it handles both sides.

    Args:
        document_type: The classified document type.

    Returns:
        An instance of the appropriate BaseExtractor subclass.

    Raises:
        ValueError: If the document type is UNKNOWN or not supported.
    """
    extractors: dict[DocumentType, type[BaseExtractor]] = {
        DocumentType.AADHAAR_FRONT: AadhaarExtractor,
        DocumentType.AADHAAR_BACK: AadhaarExtractor,
        DocumentType.PAN_CARD: PANExtractor,
        DocumentType.DRIVING_LICENSE: DrivingLicenseExtractor,
    }

    extractor_class = extractors.get(document_type)
    if extractor_class is None:
        raise ValueError(
            f"No extractor available for document type: {document_type.value}"
        )

    return extractor_class()
