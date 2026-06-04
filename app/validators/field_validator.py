"""
Field validation for extracted KYC document data.

This module validates and normalizes extracted fields such as dates, pincodes,
and license numbers according to Indian document format standards.
"""

import re

from app.models.document import DocumentType
from app.models.validation import ValidationResult


class FieldValidator:
    """Validates extracted KYC fields and normalizes their formats."""

    # Supported date format patterns with named groups
    _DATE_PATTERNS: list[tuple[str, str]] = [
        # DD/MM/YYYY (already in target format)
        (r"^(\d{1,2})/(\d{1,2})/(\d{4})$", "DMY"),
        # DD-MM-YYYY
        (r"^(\d{1,2})-(\d{1,2})-(\d{4})$", "DMY"),
        # DD.MM.YYYY
        (r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", "DMY"),
        # YYYY-MM-DD (ISO format)
        (r"^(\d{4})-(\d{1,2})-(\d{1,2})$", "YMD"),
        # YYYY/MM/DD
        (r"^(\d{4})/(\d{1,2})/(\d{1,2})$", "YMD"),
    ]

    # Indian state codes for Driving License validation (2 uppercase letters)
    _STATE_CODES = frozenset([
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA",
        "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
        "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
        "TN", "TR", "TS", "UK", "UP", "WB",
    ])

    def validate_date(self, date_str: str) -> ValidationResult:
        """Validate and normalize a date string to DD/MM/YYYY format.

        Supports the following input formats:
        - DD/MM/YYYY (no change needed)
        - DD-MM-YYYY → DD/MM/YYYY
        - DD.MM.YYYY → DD/MM/YYYY
        - YYYY-MM-DD → DD/MM/YYYY (ISO format)
        - YYYY/MM/DD → DD/MM/YYYY

        Validates day (1-31), month (1-12), year (1900-2100).

        Args:
            date_str: The date string to validate and normalize.

        Returns:
            ValidationResult with normalized DD/MM/YYYY value if valid.
        """
        if not date_str or not isinstance(date_str, str):
            return ValidationResult(
                is_valid=False,
                error_message="Date string is empty or invalid",
            )

        date_str = date_str.strip()

        for pattern, fmt in self._DATE_PATTERNS:
            match = re.match(pattern, date_str)
            if match:
                groups = match.groups()
                if fmt == "DMY":
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                else:  # YMD
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])

                # Validate ranges
                if not (1900 <= year <= 2100):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Year {year} is out of valid range (1900-2100)",
                    )
                if not (1 <= month <= 12):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Month {month} is out of valid range (1-12)",
                    )
                if not (1 <= day <= 31):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Day {day} is out of valid range (1-31)",
                    )

                # Format as DD/MM/YYYY with zero-padding
                normalized = f"{day:02d}/{month:02d}/{year}"
                return ValidationResult(
                    is_valid=True,
                    normalized_value=normalized,
                )

        return ValidationResult(
            is_valid=False,
            error_message=f"Date '{date_str}' does not match any supported format",
        )

    def validate_pincode(self, pincode: str) -> ValidationResult:
        """Validate an Indian PIN code.

        Rules:
        - Exactly 6 digits
        - First digit must be 1-9 (not 0)

        Args:
            pincode: The pincode string to validate.

        Returns:
            ValidationResult with the pincode as normalized_value if valid.
        """
        if not pincode or not isinstance(pincode, str):
            return ValidationResult(
                is_valid=False,
                error_message="Pincode is empty or invalid",
            )

        pincode = pincode.strip()

        if not re.match(r"^\d{6}$", pincode):
            return ValidationResult(
                is_valid=False,
                error_message=f"Pincode '{pincode}' must be exactly 6 digits",
            )

        if pincode[0] == "0":
            return ValidationResult(
                is_valid=False,
                error_message="Pincode first digit must be between 1 and 9",
            )

        return ValidationResult(
            is_valid=True,
            normalized_value=pincode,
        )

    def validate_license_number(self, license_no: str) -> ValidationResult:
        """Validate an Indian Driving License number format.

        Format: 2 uppercase letters (state code) followed by digits and optional
        separators (hyphens or spaces). Common formats:
        - KA01 20170001234
        - DL-0420110149646
        - MH12 20190001234

        The pattern requires:
        - 2 uppercase letters (state code)
        - Followed by digits, hyphens, or spaces (at least 4 digits total)

        Args:
            license_no: The license number string to validate.

        Returns:
            ValidationResult with the normalized license number if valid.
        """
        if not license_no or not isinstance(license_no, str):
            return ValidationResult(
                is_valid=False,
                error_message="License number is empty or invalid",
            )

        license_no = license_no.strip()

        # Pattern: 2 uppercase state code letters followed by digits/separators
        # At least 2 letters + some digits with optional separators
        pattern = r"^([A-Z]{2})[\s\-]?(\d{2})[\s\-]?(.+)$"
        match = re.match(pattern, license_no)

        if not match:
            return ValidationResult(
                is_valid=False,
                error_message=f"License number '{license_no}' does not match expected format (state-code + number)",
            )

        state_code = match.group(1)

        # Validate state code
        if state_code not in self._STATE_CODES:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid state code '{state_code}' in license number",
            )

        # Check that the remainder contains digits (with optional separators)
        remainder = match.group(2) + match.group(3)
        digits_only = re.sub(r"[\s\-]", "", remainder)
        if not digits_only.isdigit() or len(digits_only) < 4:
            return ValidationResult(
                is_valid=False,
                error_message=f"License number must contain at least 4 digits after state code",
            )

        return ValidationResult(
            is_valid=True,
            normalized_value=license_no,
        )

    def validate_all(self, document_type: DocumentType, fields: dict) -> dict:
        """Apply document-type-specific validations to extracted fields.

        Routes validation based on document type:
        - Aadhaar (front/back): validates dob, pincode
        - PAN: validates dob
        - Driving License: validates dob, issue_date, expiry_date, license_number

        Args:
            document_type: The type of document the fields were extracted from.
            fields: Dictionary of field names to extracted values.

        Returns:
            Dictionary mapping field names to their ValidationResult.
        """
        results: dict[str, ValidationResult] = {}

        if document_type in (DocumentType.AADHAAR_FRONT, DocumentType.AADHAAR_BACK):
            if "dob" in fields and fields["dob"]:
                results["dob"] = self.validate_date(fields["dob"])
            if "pincode" in fields and fields["pincode"]:
                results["pincode"] = self.validate_pincode(fields["pincode"])

        elif document_type == DocumentType.PAN_CARD:
            if "dob" in fields and fields["dob"]:
                results["dob"] = self.validate_date(fields["dob"])

        elif document_type == DocumentType.DRIVING_LICENSE:
            if "dob" in fields and fields["dob"]:
                results["dob"] = self.validate_date(fields["dob"])
            if "issue_date" in fields and fields["issue_date"]:
                results["issue_date"] = self.validate_date(fields["issue_date"])
            if "expiry_date" in fields and fields["expiry_date"]:
                results["expiry_date"] = self.validate_date(fields["expiry_date"])
            if "license_number" in fields and fields["license_number"]:
                results["license_number"] = self.validate_license_number(
                    fields["license_number"]
                )

        return results
