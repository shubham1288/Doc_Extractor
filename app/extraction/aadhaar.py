"""Aadhaar card field extractor.

Extracts structured fields from Aadhaar card OCR text including:
- Aadhaar number (masked XXXX XXXX 1234 or full 12-digit)
- Name
- Date of birth
- Gender
- Address, pincode, and state
"""

import re

from app.extraction.base import BaseExtractor, ExtractionField
from app.extraction.verhoeff import validate_verhoeff
from app.models.ocr import OCRResult

# Indian states and union territories for address extraction
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry",
    "Chandigarh", "Andaman and Nicobar", "Dadra and Nagar Haveli",
    "Daman and Diu", "Lakshadweep",
]


class AadhaarExtractor(BaseExtractor):
    """Extracts fields from Aadhaar card (front and back)."""

    # Regex for full 12-digit Aadhaar number (with optional spaces/separators)
    _AADHAAR_FULL_PATTERN = re.compile(r"(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})")

    # Regex for masked Aadhaar number (XXXX XXXX 1234 or xxxx xxxx 1234)
    _AADHAAR_MASKED_PATTERN = re.compile(
        r"([Xx]{4}[\s\-]?[Xx]{4}[\s\-]?\d{4})"
    )

    # Date patterns (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, D/M/YYYY variants)
    _DATE_PATTERN = re.compile(
        r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})"
    )

    # Gender keywords (including Hindi equivalents and OCR variants)
    _GENDER_PATTERN = re.compile(
        r"\b(Male|Female|Transgender|MALE|FEMALE|TRANSGENDER|पुरुष|महिला|स्त्री)\b",
        re.IGNORECASE,
    )

    def extract(self, ocr_result: OCRResult) -> dict[str, ExtractionField]:
        """Extract Aadhaar card fields from OCR result.

        Pre-filters the OCR text to remove instructional/footer content
        commonly found in e-Aadhaar letter PDFs.

        Extracts: aadhaar_number, name, dob, gender, address, pincode, state.

        Args:
            ocr_result: OCR output containing text and confidence scores.

        Returns:
            Dictionary mapping field names to ExtractionField instances.
        """
        text = self._filter_aadhaar_text(ocr_result.text)
        base_confidence = min(ocr_result.avg_confidence, 1.0)

        fields: dict[str, ExtractionField] = {
            "aadhaar_number": self._extract_aadhaar_number(text, base_confidence),
            "name": self._extract_name(text, base_confidence),
            "dob": self._extract_dob(text, base_confidence),
            "gender": self._extract_gender(text, base_confidence),
            "address": self._extract_address(text, base_confidence),
            "pincode": self._extract_pincode(text, base_confidence),
            "state": self._extract_state(text, base_confidence),
        }

        return fields

    def _filter_aadhaar_text(self, text: str) -> str:
        """Filter out instructional/footer text from e-Aadhaar PDFs.

        e-Aadhaar letters contain a lot of boilerplate instruction text
        that should not be parsed for fields. This removes lines containing
        known instruction patterns.

        Args:
            text: Raw OCR text.

        Returns:
            Filtered text with instructional lines removed.
        """
        noise_patterns = [
            r"this\s+(?:aadhaar|document|letter)\s+(?:should|can|is|was)",
            r"(?:should|can)\s+be\s+(?:verified|updated|used)",
            r"maadhaar|m-aadhaar|m\s*aadhaar",
            r"online\s+(?:verification|e-kyc|ekyc)",
            r"keep\s+your\s+mobile",
            r"updated\s*in\s*aadhaar",
            r"https?://",
            r"www\.",
            r"resident\.uidai",
            r"helpline|toll\s*free",
            r"qr\s*code",
            r"note\s*:",
            r"disclaimer",
            r"e-?aadhaar\s+letter",
            r"enrolment\s+(?:no|number|date|id)",
            r"vid\s*(?:no|number|:)",
            r"download\s+(?:date|time)",
        ]
        
        filtered_lines = []
        for line in text.split("\n"):
            line_lower = line.strip().lower()
            if not line_lower:
                continue
            # Skip lines matching noise patterns
            if any(re.search(p, line_lower) for p in noise_patterns):
                continue
            filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def _extract_aadhaar_number(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract Aadhaar number supporting both masked and unmasked formats.

        Distinguishes between:
        - Aadhaar number (12 digits): the actual identity number
        - VID (Virtual ID, 16 digits): NOT the Aadhaar number
        
        Tries masked format first (XXXX XXXX 1234), then full 12-digit format.
        Skips numbers that appear near "VID" keyword.
        Validates unmasked numbers using Verhoeff checksum.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted Aadhaar number.
        """
        # Try masked format first (most reliable — unique to Aadhaar)
        masked_match = self._AADHAAR_MASKED_PATTERN.search(text)
        if masked_match:
            raw = masked_match.group(1)
            # Normalize format: XXXX XXXX 1234
            digits = raw.replace(" ", "").replace("-", "")
            normalized = f"{digits[:4]} {digits[4:8]} {digits[8:]}"
            return ExtractionField(
                value=normalized,
                confidence=base_confidence * 0.9,
                raw_text=raw,
            )

        # Find ALL 12-digit number matches and pick the best one
        # Skip numbers near "VID" or "Virtual" keywords
        lines = text.split("\n")
        candidates = []
        
        for line in lines:
            line_lower = line.lower()
            # Skip lines that mention VID/Virtual ID
            if re.search(r"\bvid\b|virtual\s*id|virtual\s*number", line_lower):
                continue
            
            for match in self._AADHAAR_FULL_PATTERN.finditer(line):
                raw = match.group(1)
                digits_only = raw.replace(" ", "").replace("-", "")
                
                # Must be exactly 12 digits
                if len(digits_only) != 12:
                    continue
                
                # Skip if it's all same digits (unlikely to be real)
                if len(set(digits_only)) <= 2:
                    continue
                
                # Validate with Verhoeff checksum
                try:
                    is_valid = validate_verhoeff(digits_only)
                except ValueError:
                    is_valid = False

                score = 2 if is_valid else 0
                # Boost score if near "Aadhaar" keyword
                if re.search(r"aadhaar|uid", line_lower):
                    score += 1
                    
                candidates.append((score, digits_only, raw))

        if candidates:
            # Pick the highest-scoring candidate
            candidates.sort(key=lambda x: -x[0])
            best_digits = candidates[0][1]
            best_raw = candidates[0][2]
            is_valid = candidates[0][0] >= 2
            
            confidence = base_confidence * (0.95 if is_valid else 0.7)
            normalized = f"{best_digits[:4]} {best_digits[4:8]} {best_digits[8:]}"
            return ExtractionField(
                value=normalized,
                confidence=confidence,
                raw_text=best_raw,
            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_name(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract name from Aadhaar OCR text.

        Uses multiple strategies based on Aadhaar card layout:
        - English name typically appears AFTER the Hindi name line
        - On Aadhaar front: layout is typically Hindi name, then English name
        - Look for explicit labels first, then positional heuristics

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted name.
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # Strategy 1: Look for explicit "Name:" or similar labels
        name_keywords = [
            r"(?:^|\s)name\s*[:\-]\s*(.+)",
            r"नाम\s*[:\-]\s*(.+)",
        ]

        for line in lines:
            for keyword in name_keywords:
                match = re.search(keyword, line, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    if name and len(name) > 2 and not re.search(r"\d{4}", name):
                        normalized = self._normalize_name(name)
                        if len(normalized) > 2:
                            return ExtractionField(
                                value=normalized,
                                confidence=base_confidence * 0.9,
                                raw_text=name,
                            )

        # Strategy 2: On Aadhaar cards, the English name is typically:
        # - A line that's mostly English alphabetic characters (A-Z)
        # - NOT a known header/keyword
        # - Has at least 2 words (first name + last name) or is > 5 chars
        # - Appears before the DOB/gender line
        
        excluded_keywords = [
            "government", "india", "uidai", "male", "female", "transgender",
            "date", "birth", "address", "dob", "unique", "identification",
            "authority", "enrol", "download", "help", "vid", "your",
            "aadhaar", "aadhar", "mera", "issue", "print", "generated",
            "to", "of", "the", "is", "and", "for", "should", "verified",
            "document", "specified", "updated", "mobile", "email",
            "letter", "online", "number", "keep", "this", "that",
            "resident", "note", "disclaimer", "www", "http",
        ]

        candidates = []
        for i, line in enumerate(lines):
            # Skip empty or very short lines
            if len(line) < 4:
                continue

            # Must be predominantly English alphabetic characters
            english_chars = sum(1 for c in line if c.isascii() and c.isalpha())
            total_chars = sum(1 for c in line if not c.isspace())
            if total_chars == 0:
                continue

            english_ratio = english_chars / total_chars
            if english_ratio < 0.8:
                continue

            # Skip lines with digits (dates, numbers)
            if re.search(r"\d{2,}", line):
                continue

            # Skip known keywords
            line_lower = line.lower()
            if any(kw in line_lower for kw in excluded_keywords):
                continue

            # Skip if it's only 1 very short word
            words = line.split()
            if len(words) == 1 and len(line) < 5:
                continue

            # Skip lines with very short word fragments (OCR noise like "Pd Aq Pe")
            if all(len(w) <= 3 for w in words):
                continue

            # Good candidate — score it
            # Prefer lines with 2+ words, proper capitalization, longer words
            score = 0
            if len(words) >= 2:
                score += 2
            if line[0].isupper():
                score += 1
            if all(w[0].isupper() for w in words if w):
                score += 1
            if len(line) >= 6:
                score += 1
            # Penalize if all words are very short (OCR noise)
            avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
            if avg_word_len < 3:
                score -= 3
            # Boost if words are reasonable name-length (4+ chars)
            if avg_word_len >= 4:
                score += 2

            candidates.append((score, i, line))

        if candidates:
            # Pick the highest-scoring candidate (must have positive score)
            candidates.sort(key=lambda x: (-x[0], x[1]))
            if candidates[0][0] > 0:
                best_line = candidates[0][2]
                normalized = self._normalize_name(best_line)
                if len(normalized) > 3:
                    return ExtractionField(
                        value=normalized,
                        confidence=base_confidence * 0.7,
                        raw_text=best_line,
                    )

        # Strategy 3: Try to infer name from S/O, D/O pattern in address
        # The person's name often appears on the line BEFORE the S/O line
        # OR shares a surname with the father's name in S/O
        for i, line in enumerate(lines):
            so_match = re.search(r"[SDWC]/[Oo]\s*[:\-]?\s*(.+)", line)
            if not so_match:
                so_match = re.search(r"(?:S/O|D/O|W/O|C/O)\s*[:\-]?\s*(.+)", line)
            if so_match:
                father_name = so_match.group(1).strip().rstrip(",.")
                # Check lines above for the person's name
                for j in range(max(0, i - 3), i):
                    prev_line = lines[j].strip()
                    if not prev_line or len(prev_line) < 4:
                        continue
                    # Skip if it's a header/keyword line
                    prev_lower = prev_line.lower()
                    if any(kw in prev_lower for kw in excluded_keywords):
                        continue
                    # Check if it's mostly English text
                    english_chars = sum(1 for c in prev_line if c.isascii() and c.isalpha())
                    total = sum(1 for c in prev_line if not c.isspace())
                    if total > 0 and english_chars / total > 0.6:
                        # Check word lengths are reasonable for a name
                        words = prev_line.split()
                        if words and not all(len(w) <= 2 for w in words):
                            normalized = self._normalize_name(prev_line)
                            if len(normalized) > 3:
                                return ExtractionField(
                                    value=normalized,
                                    confidence=base_confidence * 0.65,
                                    raw_text=prev_line,
                                )
                break

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_dob(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract date of birth from Aadhaar OCR text.

        Aadhaar cards typically have multiple dates:
        - Date of Birth (what we want)
        - Enrolment date / print date (what we don't want)

        Strategy: Prioritize dates that appear near DOB keywords.
        Avoid dates near "enrolment", "print", "issue", "generated" keywords.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted DOB.
        """
        # Strategy 1: Date explicitly after DOB/Birth keywords
        dob_patterns = [
            re.compile(
                r"(?:DOB|D\.O\.B|Date\s*of\s*Birth|Birth\s*Date|जन्म\s*(?:तिथि|दिनांक)?)\s*[:\-/]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})",
                re.IGNORECASE,
            ),
            # Pattern: "Birth" keyword followed by date on same or next context
            re.compile(
                r"(?:Birth|जन्म)\s*[:\-/]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})",
                re.IGNORECASE,
            ),
        ]

        for pattern in dob_patterns:
            match = pattern.search(text)
            if match:
                return ExtractionField(
                    value=match.group(1),
                    confidence=base_confidence * 0.9,
                    raw_text=match.group(0),
                )

        # Strategy 2: Year of Birth pattern
        yob_pattern = re.compile(
            r"(?:Year\s*of\s*Birth|YOB|Birth\s*Year)\s*[:\-/]?\s*(\d{4})",
            re.IGNORECASE,
        )
        yob_match = yob_pattern.search(text)
        if yob_match:
            return ExtractionField(
                value=yob_match.group(1),
                confidence=base_confidence * 0.8,
                raw_text=yob_match.group(0),
            )

        # Strategy 3: Find all dates and pick the one most likely to be DOB
        # Exclude dates near "enrol", "print", "issue", "generated", "download"
        lines = text.split("\n")
        non_dob_keywords = ["enrol", "print", "issue", "generat", "download", "valid"]
        
        all_dates = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Skip lines with non-DOB keywords
            is_non_dob_line = any(kw in line_lower for kw in non_dob_keywords)
            
            for match in self._DATE_PATTERN.finditer(line):
                date_str = match.group(1)
                # Parse to check if it's a reasonable DOB (person would be 1-120 years old)
                try:
                    parts = re.split(r"[/\-\.]", date_str)
                    year = int(parts[2]) if len(parts[2]) == 4 else int(parts[0])
                    # A DOB year should be between ~1905 and ~2020
                    is_reasonable_dob = 1905 <= year <= 2020
                except (ValueError, IndexError):
                    is_reasonable_dob = True

                score = 0
                if is_reasonable_dob:
                    score += 2
                if not is_non_dob_line:
                    score += 2
                if re.search(r"(?:dob|birth|जन्म)", line_lower):
                    score += 5

                all_dates.append((score, date_str, match.group(0)))

        if all_dates:
            # Pick the date with the highest score
            all_dates.sort(key=lambda x: -x[0])
            best = all_dates[0]
            confidence = base_confidence * (0.8 if best[0] >= 4 else 0.5)
            return ExtractionField(
                value=best[1],
                confidence=confidence,
                raw_text=best[2],
            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_gender(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract gender from Aadhaar OCR text.

        Looks for Male/Female/Transgender keywords including Hindi equivalents.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted gender.
        """
        match = self._GENDER_PATTERN.search(text)
        if match:
            raw_gender = match.group(1)
            # Normalize to standard English values
            gender_lower = raw_gender.lower()
            if gender_lower in ("male", "पुरुष"):
                gender = "Male"
            elif gender_lower in ("female", "महिला", "स्त्री"):
                gender = "Female"
            else:
                gender = "Transgender"

            return ExtractionField(
                value=gender,
                confidence=base_confidence * 0.95,
                raw_text=match.group(0),
            )

        # Fallback: look for gender indicators with surrounding text
        gender_indicators = [
            (r"(?:gender|लिंग)\s*[:\-]?\s*(male|female|पुरुष|महिला)", re.IGNORECASE),
        ]
        for pattern, flags in gender_indicators:
            m = re.search(pattern, text, flags)
            if m:
                val = m.group(1).lower()
                if val in ("male", "पुरुष"):
                    gender = "Male"
                else:
                    gender = "Female"
                return ExtractionField(
                    value=gender,
                    confidence=base_confidence * 0.85,
                    raw_text=m.group(0),
                )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_address(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract address from Aadhaar back card OCR text.

        Looks for address section after address keywords, or infers address
        from common address indicators (S/O, D/O, house number patterns).

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted address.
        """
        lines = text.split("\n")

        # Find address start - look for address keyword or common address indicators
        address_start = -1
        for i, line in enumerate(lines):
            if re.search(r"(?:address|पता|addr)\s*[:\-]?", line, re.IGNORECASE):
                address_start = i
                break

        # If no explicit "Address" keyword, look for S/O, D/O, W/O, C/O patterns
        if address_start == -1:
            for i, line in enumerate(lines):
                if re.search(r"\b[SDWC]/[Oo]\b|(?:S/O|D/O|W/O|C/O)", line):
                    address_start = i
                    break

        if address_start == -1:
            return ExtractionField(value=None, confidence=0.0, raw_text=None)

        # Collect address lines (typically 2-6 lines after the keyword)
        address_parts: list[str] = []
        # Check if there's content on the same line as the keyword
        keyword_match = re.search(
            r"(?:address|पता|addr)\s*[:\-]?\s*(.*)", lines[address_start], re.IGNORECASE
        )
        if keyword_match and keyword_match.group(1).strip():
            address_parts.append(keyword_match.group(1).strip())
        elif not re.search(r"(?:address|पता|addr)", lines[address_start], re.IGNORECASE):
            # Line starts with S/O etc, include the whole line
            address_parts.append(lines[address_start].strip())

        for line in lines[address_start + 1 : address_start + 7]:
            stripped = line.strip()
            if not stripped:
                break
            # Stop at next field label or Aadhaar number
            if re.match(r"^(VID|UID|Aadhaar|Help|\d{4}\s?\d{4}\s?\d{4})", stripped, re.IGNORECASE):
                break
            # Stop if we hit a line that's clearly not address (e.g. website)
            if re.search(r"(www\.|\.gov\.|\.com|helpline)", stripped, re.IGNORECASE):
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

    def _extract_pincode(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract 6-digit Indian PIN code from text.

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted pincode.
        """
        # Look for pincode pattern (6 digits, first digit 1-9)
        pincode_pattern = re.compile(r"\b([1-9]\d{5})\b")

        # Prefer pincode near "pin" keyword
        pin_keyword = re.compile(
            r"(?:pin\s*code|pin)\s*[:\-]?\s*(\d{6})", re.IGNORECASE
        )
        match = pin_keyword.search(text)
        if match:
            return ExtractionField(
                value=match.group(1),
                confidence=base_confidence * 0.9,
                raw_text=match.group(0),
            )

        # Fallback: find any 6-digit number that looks like a pincode
        all_matches = pincode_pattern.findall(text)
        if all_matches:
            # Take the last one (usually at end of address)
            pincode = all_matches[-1]
            return ExtractionField(
                value=pincode,
                confidence=base_confidence * 0.7,
                raw_text=pincode,
            )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)

    def _extract_state(self, text: str, base_confidence: float) -> ExtractionField:
        """Extract Indian state name from text.

        Uses multiple strategies:
        1. Look for state name near "State:" or "District:" keywords
        2. Use pincode first-2-digits to validate/determine state
        3. Look for exact state name matches (with word boundaries)

        Args:
            text: OCR text to search.
            base_confidence: Base confidence from OCR.

        Returns:
            ExtractionField with the extracted state.
        """
        # Pincode-to-state mapping (first 1-2 digits of pincode → state)
        pincode_state_map = {
            "11": "Delhi", "12": "Haryana", "13": "Haryana",
            "14": "Punjab", "15": "Punjab", "16": "Punjab",
            "17": "Himachal Pradesh", "18": "Jammu and Kashmir",
            "19": "Jammu and Kashmir",
            "20": "Uttar Pradesh", "21": "Uttar Pradesh",
            "22": "Uttar Pradesh", "23": "Uttar Pradesh",
            "24": "Uttar Pradesh", "25": "Uttar Pradesh",
            "26": "Uttarakhand", "27": "Uttar Pradesh",
            "28": "Uttar Pradesh",
            "30": "Rajasthan", "31": "Rajasthan", "32": "Rajasthan",
            "33": "Rajasthan", "34": "Rajasthan",
            "36": "Haryana",
            "37": "Gujarat", "38": "Gujarat", "39": "Gujarat",
            "40": "Maharashtra", "41": "Maharashtra", "42": "Maharashtra",
            "43": "Maharashtra", "44": "Maharashtra", "45": "Madhya Pradesh",
            "46": "Madhya Pradesh", "47": "Madhya Pradesh", "48": "Madhya Pradesh",
            "49": "Chhattisgarh",
            "50": "Telangana", "51": "Telangana", "52": "Andhra Pradesh",
            "53": "Andhra Pradesh", "54": "Andhra Pradesh", "55": "Andhra Pradesh",
            "56": "Karnataka", "57": "Karnataka", "58": "Karnataka", "59": "Karnataka",
            "60": "Tamil Nadu", "61": "Tamil Nadu", "62": "Tamil Nadu", "63": "Tamil Nadu",
            "64": "Tamil Nadu",
            "67": "Kerala", "68": "Kerala", "69": "Kerala",
            "70": "West Bengal", "71": "West Bengal", "72": "West Bengal",
            "73": "West Bengal", "74": "West Bengal",
            "75": "Odisha", "76": "Odisha", "77": "Odisha",
            "78": "Assam", "79": "Assam",
            "80": "Bihar", "81": "Bihar", "82": "Bihar", "83": "Bihar", "84": "Bihar",
            "85": "Bihar",
            "82": "Jharkhand", "83": "Jharkhand",
            "90": "Manipur", "91": "Mizoram", "92": "Tripura",
            "79": "Meghalaya", "78": "Nagaland",
        }

        # Strategy 1: Try to determine state from pincode
        pincode_pattern = re.compile(r"\b([1-9]\d{5})\b")
        pincodes = pincode_pattern.findall(text)
        if pincodes:
            # Use the last pincode found (usually in address)
            pin = pincodes[-1]
            first_two = pin[:2]
            if first_two in pincode_state_map:
                state = pincode_state_map[first_two]
                return ExtractionField(
                    value=state,
                    confidence=base_confidence * 0.8,
                    raw_text=f"pincode:{pin}",
                )

        # Strategy 2: Look for state near "State:" keyword
        state_keyword_pattern = re.compile(
            r"(?:state|राज्य)\s*[:\-]?\s*(.+)", re.IGNORECASE
        )
        match = state_keyword_pattern.search(text)
        if match:
            state_text = match.group(1).strip()
            # Try to match against known states
            for state in INDIAN_STATES:
                if state.lower() in state_text.lower():
                    return ExtractionField(
                        value=state,
                        confidence=base_confidence * 0.85,
                        raw_text=state_text,
                    )

        # Strategy 3: Look for exact state name with word boundaries
        text_lower = text.lower()
        for state in INDIAN_STATES:
            # Require word boundary or start/end of line
            pattern = r"\b" + re.escape(state.lower()) + r"\b"
            if re.search(pattern, text_lower):
                return ExtractionField(
                    value=state,
                    confidence=base_confidence * 0.7,
                    raw_text=state,
                )

        return ExtractionField(value=None, confidence=0.0, raw_text=None)
