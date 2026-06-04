"""Field extraction engines for KYC documents.

Provides document-specific extractors for Aadhaar, PAN, and Driving License,
along with a router to select the appropriate extractor based on document type.
"""

from app.extraction.aadhaar import AadhaarExtractor
from app.extraction.base import BaseExtractor, ExtractionField
from app.extraction.driving_license import DrivingLicenseExtractor
from app.extraction.pan import PANExtractor
from app.extraction.router import get_extractor
from app.extraction.verhoeff import validate_verhoeff

__all__ = [
    "AadhaarExtractor",
    "BaseExtractor",
    "DrivingLicenseExtractor",
    "ExtractionField",
    "PANExtractor",
    "get_extractor",
    "validate_verhoeff",
]
