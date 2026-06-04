"""Base extractor abstract class and shared extraction data models."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.models.ocr import OCRResult


@dataclass
class ExtractionField:
    """A single extracted field with value, confidence, and raw source text.

    Attributes:
        value: The extracted and normalized field value, or None if not found.
        confidence: Confidence score between 0.0 and 1.0.
        raw_text: The raw OCR text from which the value was extracted.
    """

    value: str | None
    confidence: float
    raw_text: str | None = None


class BaseExtractor(ABC):
    """Abstract base for document-specific field extractors.

    Provides shared helpers for fuzzy string matching and Indian name
    normalization that all concrete extractors can leverage.
    """

    @abstractmethod
    def extract(self, ocr_result: OCRResult) -> dict[str, ExtractionField]:
        """Extract structured fields from OCR result.

        Args:
            ocr_result: The OCR output containing text and confidence scores.

        Returns:
            Dictionary mapping field names to ExtractionField instances.
        """
        ...

    def _fuzzy_match(self, text: str, pattern: str, threshold: float = 0.8) -> str | None:
        """Fuzzy string matching for noisy OCR text.

        Slides a window of pattern length over the text and returns the
        best matching substring if its similarity ratio meets the threshold.

        Args:
            text: The text to search within.
            pattern: The pattern to match against.
            threshold: Minimum similarity ratio (0.0 to 1.0) to accept a match.

        Returns:
            The best matching substring, or None if no match meets threshold.
        """
        if not text or not pattern:
            return None

        pattern_lower = pattern.lower()
        text_lower = text.lower()

        # Try exact substring first
        if pattern_lower in text_lower:
            start = text_lower.index(pattern_lower)
            return text[start : start + len(pattern)]

        # Sliding window fuzzy match
        best_match: str | None = None
        best_ratio = 0.0
        window_size = len(pattern)

        for i in range(len(text) - window_size + 1):
            window = text_lower[i : i + window_size]
            ratio = SequenceMatcher(None, pattern_lower, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = text[i : i + window_size]

        if best_ratio >= threshold:
            return best_match
        return None

    def _normalize_name(self, name: str) -> str:
        """Normalize Indian names: title case, strip extra whitespace, handle initials.

        Handles common patterns in Indian names:
        - Converts to title case
        - Strips leading/trailing whitespace
        - Collapses multiple spaces to single space
        - Preserves single-letter initials with dot (e.g., "S." stays "S.")
        - Removes common OCR artifacts

        Args:
            name: The raw name string from OCR.

        Returns:
            Normalized name string.
        """
        if not name:
            return ""

        # Remove common OCR artifacts
        name = re.sub(r"[|_\[\]{}]", "", name)

        # Strip and collapse whitespace
        name = re.sub(r"\s+", " ", name.strip())

        # Split into parts and normalize each
        parts = name.split()
        normalized_parts: list[str] = []

        for part in parts:
            # Keep initials (single letter or letter+dot) as uppercase
            if len(part) == 1 or (len(part) == 2 and part.endswith(".")):
                normalized_parts.append(part.upper())
            else:
                normalized_parts.append(part.title())

        return " ".join(normalized_parts)
