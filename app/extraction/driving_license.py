"""Driving License field extractor.

Extracts structured fields from Indian Driving License OCR text including:
- License number (various state formats)
- Name
- Date of birth
- Issue date
- Expiry date
- Address
- Issuing authority
"""

import re

from app.extraction.base import BaseExtractor, ExtractionField
from app.models.ocr import OCRResult


class DrivingLicenseExtractor(BaseExtractor):
    """Extracts fields from Indian Driving License."""

    # DL number patterns for various state formats
    # Common format: XX-YYZZZZZZZZZZ or XX/YY/ZZZZZZZZ or XXYY YYYYNNNNNNN
    _DL_PATTERNS = [
        # Format: XX00 YYYYNNNNNNN (e.g., MH02 20170012345)
        re.compile(r"\b([A-Z]{2}\d{2}\s?\d{11})\b"),
        # Format: XX-00YYYYNNNNNNN (e.g., KA-0120170012345)
        re.compile(r"\b([A-Z]{2}[\-/]\d{2}\s?\d{4}\s?\d{7})\b"),
        # Format: XX00 YYYY NNNNNNN (e.g., DL05 2019 0012345)
        re.compile(r"\b([A-Z]{2}\d{2}\s?\d{4}\s?\d{7})\b"),
        # Generic: 2 letters followed by digits and possible separators
        re.compile(r"\b([A-Z]{2}[\-/\s]?\d{2}[\-/\s]?\d{4,}[\-/\s]?\d{4,})\b"),
    ]

    # Date patterns
    _DATE_PATTERN = re.compile(r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})")

    def extract(self, ocr_result: OCRResult) -> dict[str, ExtractionField]:
        """Extract Driving License fields from OCR result.

        Extracts: license_number, name, dob, issue_date, expiry_date,
                  address, issuing_authority.

        Args:
            ocr_result: OCR output containing text and confidence scores.

        Returns:
            Dictionary mapping field names to ExtractionField instances.
        """
        text = ocr_result.text
        base_confidence = min(ocr_result.avg_confidence, 1.0)

        fields: dict[str, ExtractionField] = {
            "license_number": self._extract_license_number(text, base_confidence),
            "name": self._extract_name(text, base_confidence),
            "dob": self._extract_dob(text, base_confidence),
            "issue_date": self._extract_issue_date(text, base_confidence),
            "expiry_date": self._extract_expiry_date(text, base_confidence),
            "address": self._extract_address(text, base_confidence),
            "issuing_authority": self._extract_issuing_authority(text, base_confidence),
        }

        return fields

    def _extract_license_number(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract driving license number from various state formats.

        Indian DL numbers typically have a 2-letter state code followed
        by digits in various formats depending on the issuing state.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted license number.
        """
        # Look for DL number near relevant keywords first
        dl_keyword_pattern = re.compile(
            r"(?:DL\s*(?:No|Number)|Licence\s*No|License\s*No|DL\.?\s*No\.?)\s*[:\-]?\s*([A-Z]{2}[\-/\s]?\d{2}[\-/\s]?\d{4,}[\-/\s]?\d{3,})",
            re.IGNORECASE,
        )
        match = dl_keyword_pattern.search(text)
        if match:
            dl_number = match.group(1).replace(" ", "")
            return ExtractionField(
                value=dl_number,
                confidence=base_confidence * 0.9,
                raw_text=match.group(0),
            )

        # Try each DL pattern
        for pattern in self._DL_PATTERNS:
            match = pattern.search(text)
            if match:
                dl_number = match.group(1).replace(" ", "")
                return ExtractionField(
                    value=dl_number,
                    confidence=base_confidence * 0.8,
                    raw_text=match.group(0),
                )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_name(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract name from Driving License OCR text.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted name.
        """
        lines = text.split("\n")

        name_keywords = [
            r"name\s*[:\-]?\s*",
            r"holder'?s?\s*name\s*[:\-]?\s*",
            r"नाम\s*[:\-]?\s*",
        ]

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

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_dob(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract date of birth from Driving License OCR text.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted DOB.
        """
        dob_pattern = re.compile(
            r"(?:DOB|D\.O\.B|Date\s*of\s*Birth)\s*[:\-]?\s*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
            re.IGNORECASE,
        )
        match = dob_pattern.search(text)
        if match:
            return ExtractionField(
                value=match.group(1),
                confidence=base_confidence * 0.9,
                raw_text=match.group(0),
            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_issue_date(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract issue date from Driving License OCR text.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted issue date.
        """
        issue_pattern = re.compile(
            r"(?:Issue\s*Date|Date\s*of\s*Issue|DOI|Issued?\s*on)\s*[:\-]?\s*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
            re.IGNORECASE,
        )
        match = issue_pattern.search(text)
        if match:
            return ExtractionField(
                value=match.group(1),
                confidence=base_confidence * 0.85,
                raw_text=match.group(0),
            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_expiry_date(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract expiry date from Driving License OCR text.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted expiry date.
        """
        expiry_pattern = re.compile(
            r"(?:Expiry\s*Date|Valid\s*(?:Till|Until|Upto)|Expir(?:y|es)|Validity)\s*[:\-]?\s*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",
            re.IGNORECASE,
        )
        match = expiry_pattern.search(text)
        if match:
            return ExtractionField(
                value=match.group(1),
                confidence=base_confidence * 0.85,
                raw_text=match.group(0),
            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_address(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract address from Driving License OCR text.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted address.
        """
        lines = text.split("\n")

        # Find address start
        address_start = -1
        for i, line in enumerate(lines):
            if re.search(r"(?:address|पता)\s*[:\-]?", line, re.IGNORECASE):
                address_start = i
                break

        if address_start == -1:
            return ExtractionField(value=None, confidence=0.0, raw_text=None)

        # Collect address lines
        address_parts: list[str] = []
        keyword_match = re.search(
            r"(?:address|पता)\s*[:\-]?\s*(.*)", lines[address_start], re.IGNORECASE
        )
        if keyword_match and keyword_match.group(1).strip():
            address_parts.append(keyword_match.group(1).strip())

        for line in lines[address_start + 1 : address_start + 5]:
            stripped = line.strip()
            if not stripped:
                break
            # Stop at next field label
            if re.match(
                r"^(Blood|Class|Validity|Expiry|Issue|DOB|Name)",
                stripped,
                re.IGNORECASE,
            ):
                break
            address_parts.append(stripped)

        if address_parts:
            address = ", ".join(address_parts)
            return ExtractionField(
                value=address,
                confidence=base_confidence * 0.75,
                raw_text="\n".join(address_parts),
            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_issuing_authority(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract issuing authority (RTO) from Driving License OCR text.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted issuing authority.
        """
        authority_pattern = re.compile(
            r"(?:Issuing\s*Authority|Issued?\s*by|RTO|Regional\s*Transport)\s*[:\-]?\s*(.+)",
            re.IGNORECASE,
        )
        match = authority_pattern.search(text)
        if match:
            authority = match.group(1).strip()
            if authority:
                return ExtractionField(
                    value=authority,
                    confidence=base_confidence * 0.8,
                    raw_text=match.group(0),
                )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)
