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

        PAN card layout (top to bottom):
        - Header (Income Tax / Govt of India)
        - PAN number
        - Name (English, ALL CAPS)
        - Hindi name (Devanagari)
        - Father's name label + name
        - DOB

        Strategy: Find name by position relative to PAN number, or by
        looking for uppercase English lines that aren't headers.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted name.
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # Strategy 1: Look for explicit "Name:" label
        for i, line in enumerate(lines):
            match = re.search(r"(?:^|\s)name\s*[:\-]\s*(.+)", line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if name and len(name) > 2 and not re.search(r"\d{3,}", name):
                    normalized = self._normalize_name(name)
                    if len(normalized) > 2:
                        return ExtractionField(
                            value=normalized,
                            confidence=base_confidence * 0.9,
                            raw_text=name,
                        )

        # Strategy 2: Find PAN number line, then look at nearby lines
        pan_line_idx = -1
        for i, line in enumerate(lines):
            if self._PAN_PATTERN.search(line):
                pan_line_idx = i
                break

        # On PAN cards, the English name is typically 1-3 lines after PAN number
        if pan_line_idx >= 0:
            for offset in range(1, 4):
                idx = pan_line_idx + offset
                if idx >= len(lines):
                    break
                candidate = lines[idx]
                # Must be mostly English uppercase and not a keyword
                english_upper = sum(1 for c in candidate if c.isupper() and c.isascii())
                total_alpha = sum(1 for c in candidate if c.isalpha())
                
                if total_alpha < 3:
                    continue
                if english_upper / max(total_alpha, 1) < 0.5:
                    continue
                if any(kw in candidate.lower() for kw in [
                    "income", "tax", "india", "permanent", "account",
                    "govt", "department", "father", "signature",
                ]):
                    continue
                if re.search(r"\d{3,}", candidate):
                    continue
                    
                normalized = self._normalize_name(candidate)
                if len(normalized) > 2:
                    return ExtractionField(
                        value=normalized,
                        confidence=base_confidence * 0.75,
                        raw_text=candidate,
                    )

        # Strategy 3: Heuristic - look for all-caps English name lines
        excluded = [
            "income", "tax", "india", "permanent", "account",
            "govt", "department", "father", "signature", "date",
            "birth", "dob",
        ]
        
        candidates = []
        for i, line in enumerate(lines):
            # Must have at least some English uppercase letters
            english_upper = sum(1 for c in line if c.isupper() and c.isascii())
            total_chars = sum(1 for c in line if not c.isspace())
            
            if total_chars < 3 or english_upper < 3:
                continue
            if english_upper / max(total_chars, 1) < 0.6:
                continue
            if any(kw in line.lower() for kw in excluded):
                continue
            if re.search(r"\d{3,}", line):
                continue
            
            # Score: prefer longer names, more uppercase
            words = line.split()
            score = english_upper + len(words) * 2
            candidates.append((score, i, line))

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            best = candidates[0][2]
            normalized = self._normalize_name(best)
            if len(normalized) > 2:
                return ExtractionField(
                    value=normalized,
                    confidence=base_confidence * 0.65,
                    raw_text=best,
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
