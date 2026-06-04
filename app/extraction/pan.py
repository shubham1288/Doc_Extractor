"""PAN card field extractor.

Extracts structured fields from PAN card OCR text including:
- PAN number (ABCDE1234F format)
- Name
- Father's name
- Date of birth
"""

import re

from app.extraction.base import BaseExtractor, ExtractionField
from app.models.ocr import OCRResult


class PANExtractor(BaseExtractor):
    """Extracts fields from PAN card."""

    # PAN number format: 5 uppercase letters + 4 digits + 1 uppercase letter
    _PAN_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")

    # Date patterns (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY)
    _DATE_PATTERN = re.compile(r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})")

    def extract(self, ocr_result: OCRResult) -> dict[str, ExtractionField]:
        """Extract PAN card fields from OCR result.

        Extracts: pan_number, name, father_name, dob.

        Args:
            ocr_result: OCR output containing text and confidence scores.

        Returns:
            Dictionary mapping field names to ExtractionField instances.
        """
        text = ocr_result.text
        base_confidence = min(ocr_result.avg_confidence, 1.0)

        fields: dict[str, ExtractionField] = {
            "pan_number": self._extract_pan_number(text, base_confidence),
            "name": self._extract_name(text, base_confidence),
            "father_name": self._extract_father_name(text, base_confidence),
            "dob": self._extract_dob(text, base_confidence),
        }

        return fields

    def _extract_pan_number(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract PAN number in ABCDE1234F format.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted PAN number.
        """
        match = self._PAN_PATTERN.search(text)
        if match:
            pan = match.group(1)
            return ExtractionField(
                value=pan,
                confidence=base_confidence * 0.95,
                raw_text=match.group(0),
            )

        # Try case-insensitive then uppercase result
        insensitive_pattern = re.compile(r"\b([A-Za-z]{5}[0-9]{4}[A-Za-z])\b")
        match = insensitive_pattern.search(text)
        if match:
            pan = match.group(1).upper()
            return ExtractionField(
                value=pan,
                confidence=base_confidence * 0.75,
                raw_text=match.group(0),
            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_name(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract cardholder name from PAN card OCR text.

        On PAN cards, the name typically appears after 'Name' label
        or is the line after the PAN number.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted name.
        """
        lines = text.split("\n")

        # Look for explicit name label
        name_keywords = [r"name\s*[:\-]?\s*", r"नाम\s*[:\-]?\s*"]
        for i, line in enumerate(lines):
            for keyword in name_keywords:
                match = re.search(keyword, line, re.IGNORECASE)
                if match:
                    name = line[match.end():].strip()
                    if name and len(name) > 1:
                        normalized = self._normalize_name(name)
                        return ExtractionField(
                            value=normalized,
                            confidence=base_confidence * 0.85,
                            raw_text=name,
                        )

        # Heuristic: look for all-caps name lines (PAN cards use uppercase)
        for line in lines:
            stripped = line.strip()
            if (
                stripped
                and len(stripped) > 3
                and re.match(r"^[A-Z][A-Za-z\s]+$", stripped)
                and not re.search(r"\d", stripped)
                and not any(kw in stripped.lower() for kw in [
                    "income", "tax", "india", "permanent", "account",
                    "govt", "department", "father",
                ])
            ):
                normalized = self._normalize_name(stripped)
                return ExtractionField(
                    value=normalized,
                    confidence=base_confidence * 0.7,
                    raw_text=stripped,
                )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_father_name(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract father's name from PAN card OCR text.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted father's name.
        """
        lines = text.split("\n")

        # Look for father's name after keyword
        father_keywords = [
            r"father'?s?\s*name\s*[:\-]?\s*",
            r"पिता\s*(?:का\s*)?नाम\s*[:\-]?\s*",
        ]

        for i, line in enumerate(lines):
            for keyword in father_keywords:
                match = re.search(keyword, line, re.IGNORECASE)
                if match:
                    name = line[match.end():].strip()
                    if name and len(name) > 1:
                        normalized = self._normalize_name(name)
                        return ExtractionField(
                            value=normalized,
                            confidence=base_confidence * 0.85,
                            raw_text=name,
                        )
                    # Check next line if label is on its own line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and len(next_line) > 1:
                            normalized = self._normalize_name(next_line)
                            return ExtractionField(
                                value=normalized,
                                confidence=base_confidence * 0.75,
                                raw_text=next_line,
                            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_dob(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract date of birth from PAN card OCR text.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted DOB.
        """
        # Look for date after DOB keywords
        dob_pattern = re.compile(
            r"(?:DOB|D\.O\.B|Date\s*of\s*Birth|जन्म\s*तिथि)\s*[:\-]?\s*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
            re.IGNORECASE,
        )
        match = dob_pattern.search(text)
        if match:
            return ExtractionField(
                value=match.group(1),
                confidence=base_confidence * 0.9,
                raw_text=match.group(0),
            )

        # Fallback: first date pattern
        date_match = self._DATE_PATTERN.search(text)
        if date_match:
            return ExtractionField(
                value=date_match.group(1),
                confidence=base_confidence * 0.6,
                raw_text=date_match.group(0),
            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)
