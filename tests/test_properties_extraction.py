"""
Property-based tests for Extraction Engines — Verhoeff checksum, PAN format,
and extraction confidence bounds.

Uses Hypothesis to verify that extraction properties hold for ALL generated inputs.

**Validates: Requirements 5.4, 6.2, 5.5, 6.3, 7.3**
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.extraction.aadhaar import AadhaarExtractor
from app.extraction.driving_license import DrivingLicenseExtractor
from app.extraction.pan import PANExtractor
from app.extraction.verhoeff import validate_verhoeff
from app.models.ocr import OCRResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: generate a random digit position (0..11) in a 12-digit number
digit_position = st.integers(min_value=0, max_value=11)

# Strategy: generate a digit different from the original (for mutation)
digit_value = st.integers(min_value=0, max_value=9)

# Strategy: generate adjacent pair indices (0..10 means swap positions i and i+1)
adjacent_pair_index = st.integers(min_value=0, max_value=10)

# Strategy: generate valid PAN-format strings
pan_strategy = st.from_regex(r"[A-Z]{5}[0-9]{4}[A-Z]", fullmatch=True)

# Strategy: text that does NOT contain a PAN pattern
# Generate text from lowercase letters, digits, and spaces only
non_pan_text_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?()-"),
    min_size=0,
    max_size=200,
)

# Strategy: OCR confidence between 0.0 and 1.0
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy: random text content for OCR results
random_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=300,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KNOWN_VALID_NUMBER = "499118665246"


def _make_ocr_result(text: str, confidence: float) -> OCRResult:
    """Create an OCRResult with the given text and confidence."""
    return OCRResult(
        text=text,
        boxes=[],
        confidences=[confidence] if confidence > 0 else [],
        avg_confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Property 12: Verhoeff checksum correctness
# ---------------------------------------------------------------------------


class TestVerhoeffChecksumCorrectnessProperty:
    """Property 12: Verhoeff checksum correctness.

    For ANY valid 12-digit number that passes validate_verhoeff():
    - Changing ANY single digit must make it invalid
    - Swapping ANY pair of adjacent digits must make it invalid
    - validate_verhoeff is deterministic: same input always produces same output

    **Validates: Requirements 5.4**
    """

    @given(position=digit_position, new_digit=digit_value)
    @settings(max_examples=200, deadline=2000)
    def test_single_digit_mutation_invalidates(self, position: int, new_digit: int) -> None:
        """For a known valid number, changing any single digit makes it invalid."""
        original_digit = int(KNOWN_VALID_NUMBER[position])
        assume(new_digit != original_digit)

        # Mutate the digit at the given position
        mutated = list(KNOWN_VALID_NUMBER)
        mutated[position] = str(new_digit)
        mutated_number = "".join(mutated)

        assert validate_verhoeff(mutated_number) is False, (
            f"Mutating position {position} from {original_digit} to {new_digit} "
            f"in {KNOWN_VALID_NUMBER} produced {mutated_number} which should be invalid"
        )

    @given(position=adjacent_pair_index)
    @settings(max_examples=200, deadline=2000)
    def test_adjacent_transposition_invalidates(self, position: int) -> None:
        """For a known valid number, swapping any adjacent pair makes it invalid."""
        # Skip if adjacent digits are the same (swap would produce same number)
        assume(KNOWN_VALID_NUMBER[position] != KNOWN_VALID_NUMBER[position + 1])

        # Swap adjacent digits at position and position+1
        swapped = list(KNOWN_VALID_NUMBER)
        swapped[position], swapped[position + 1] = swapped[position + 1], swapped[position]
        swapped_number = "".join(swapped)

        assert validate_verhoeff(swapped_number) is False, (
            f"Swapping positions {position} and {position + 1} "
            f"in {KNOWN_VALID_NUMBER} produced {swapped_number} which should be invalid"
        )

    @given(st.just(KNOWN_VALID_NUMBER))
    @settings(max_examples=50, deadline=2000)
    def test_deterministic_valid(self, number: str) -> None:
        """validate_verhoeff is deterministic: same valid input always returns True."""
        assert validate_verhoeff(number) is True

    @given(position=digit_position, new_digit=digit_value)
    @settings(max_examples=100, deadline=2000)
    def test_deterministic_invalid(self, position: int, new_digit: int) -> None:
        """validate_verhoeff is deterministic: same invalid input always returns False."""
        original_digit = int(KNOWN_VALID_NUMBER[position])
        assume(new_digit != original_digit)

        mutated = list(KNOWN_VALID_NUMBER)
        mutated[position] = str(new_digit)
        mutated_number = "".join(mutated)

        # Call twice to verify determinism
        result1 = validate_verhoeff(mutated_number)
        result2 = validate_verhoeff(mutated_number)
        assert result1 == result2, (
            f"Non-deterministic result for {mutated_number}: "
            f"first call={result1}, second call={result2}"
        )


# ---------------------------------------------------------------------------
# Property 13: PAN format validation
# ---------------------------------------------------------------------------


class TestPANFormatValidationProperty:
    """Property 13: PAN format validation.

    For ANY string matching the PAN regex [A-Z]{5}[0-9]{4}[A-Z]:
    - PANExtractor should extract it as pan_number
    For ANY text without a PAN pattern:
    - pan_number.value should be None

    **Validates: Requirements 6.2**
    """

    @given(pan=pan_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_valid_pan_format_is_extracted(self, pan: str) -> None:
        """For any valid PAN-format string, extractor finds it as pan_number."""
        text = f"Some document text\n{pan}\nMore text"
        ocr_result = _make_ocr_result(text, confidence=0.9)
        extractor = PANExtractor()

        result = extractor.extract(ocr_result)

        assert result["pan_number"].value == pan, (
            f"Expected pan_number={pan!r} but got {result['pan_number'].value!r}"
        )

    @given(pan=pan_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_valid_pan_has_positive_confidence(self, pan: str) -> None:
        """For any valid PAN-format string, confidence is positive."""
        text = f"PAN: {pan}"
        ocr_result = _make_ocr_result(text, confidence=0.9)
        extractor = PANExtractor()

        result = extractor.extract(ocr_result)

        assert result["pan_number"].confidence > 0.0, (
            f"Expected positive confidence for PAN {pan!r} "
            f"but got {result['pan_number'].confidence}"
        )

    @given(text=non_pan_text_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_no_pan_pattern_returns_none(self, text: str) -> None:
        """For any text without PAN pattern, pan_number.value is None."""
        # Ensure text really has no PAN pattern (uppercase 5 letters + 4 digits + 1 letter)
        import re
        assume(not re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text.upper()))

        ocr_result = _make_ocr_result(text, confidence=0.9)
        extractor = PANExtractor()

        result = extractor.extract(ocr_result)

        assert result["pan_number"].value is None, (
            f"Expected None for text without PAN pattern but got "
            f"{result['pan_number'].value!r} from text: {text!r}"
        )


# ---------------------------------------------------------------------------
# Property 14: Extraction confidence bounds
# ---------------------------------------------------------------------------


class TestExtractionConfidenceBoundsProperty:
    """Property 14: Extraction confidence scores bounds.

    For ANY OCRResult with avg_confidence between 0.0 and 1.0:
    - ALL extraction fields from ANY extractor must have confidence in [0.0, 1.0]

    Tests with AadhaarExtractor, PANExtractor, and DrivingLicenseExtractor.

    **Validates: Requirements 5.5, 6.3, 7.3**
    """

    @given(text=random_text_strategy, confidence=confidence_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_aadhaar_confidence_bounds(self, text: str, confidence: float) -> None:
        """AadhaarExtractor always produces field confidences in [0.0, 1.0]."""
        ocr_result = _make_ocr_result(text, confidence)
        extractor = AadhaarExtractor()

        result = extractor.extract(ocr_result)

        for field_name, field in result.items():
            assert 0.0 <= field.confidence <= 1.0, (
                f"AadhaarExtractor field '{field_name}' confidence "
                f"{field.confidence} is outside [0.0, 1.0] "
                f"with avg_confidence={confidence}, text={text!r}"
            )

    @given(text=random_text_strategy, confidence=confidence_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_pan_confidence_bounds(self, text: str, confidence: float) -> None:
        """PANExtractor always produces field confidences in [0.0, 1.0]."""
        ocr_result = _make_ocr_result(text, confidence)
        extractor = PANExtractor()

        result = extractor.extract(ocr_result)

        for field_name, field in result.items():
            assert 0.0 <= field.confidence <= 1.0, (
                f"PANExtractor field '{field_name}' confidence "
                f"{field.confidence} is outside [0.0, 1.0] "
                f"with avg_confidence={confidence}, text={text!r}"
            )

    @given(text=random_text_strategy, confidence=confidence_strategy)
    @settings(max_examples=200, deadline=2000)
    def test_driving_license_confidence_bounds(self, text: str, confidence: float) -> None:
        """DrivingLicenseExtractor always produces field confidences in [0.0, 1.0]."""
        ocr_result = _make_ocr_result(text, confidence)
        extractor = DrivingLicenseExtractor()

        result = extractor.extract(ocr_result)

        for field_name, field in result.items():
            assert 0.0 <= field.confidence <= 1.0, (
                f"DrivingLicenseExtractor field '{field_name}' confidence "
                f"{field.confidence} is outside [0.0, 1.0] "
                f"with avg_confidence={confidence}, text={text!r}"
            )

    @given(confidence=confidence_strategy)
    @settings(max_examples=100, deadline=2000)
    def test_aadhaar_with_realistic_text_confidence_bounds(self, confidence: float) -> None:
        """AadhaarExtractor with realistic document text maintains confidence bounds."""
        text = "Government of India\nXXXX XXXX 4567\nName: RAJESH KUMAR\nDOB: 15/08/1990\nMale\nAddress: 123 Main Road\nMumbai, Maharashtra\nPin Code: 400001"
        ocr_result = _make_ocr_result(text, confidence)
        extractor = AadhaarExtractor()

        result = extractor.extract(ocr_result)

        for field_name, field in result.items():
            assert 0.0 <= field.confidence <= 1.0, (
                f"AadhaarExtractor field '{field_name}' confidence "
                f"{field.confidence} is outside [0.0, 1.0] "
                f"with avg_confidence={confidence}"
            )

    @given(confidence=confidence_strategy)
    @settings(max_examples=100, deadline=2000)
    def test_pan_with_realistic_text_confidence_bounds(self, confidence: float) -> None:
        """PANExtractor with realistic document text maintains confidence bounds."""
        text = "INCOME TAX DEPARTMENT\nPermanent Account Number\nABCDE1234F\nName: RAJESH KUMAR\nFather's Name: MOHAN KUMAR\nDOB: 15/08/1990"
        ocr_result = _make_ocr_result(text, confidence)
        extractor = PANExtractor()

        result = extractor.extract(ocr_result)

        for field_name, field in result.items():
            assert 0.0 <= field.confidence <= 1.0, (
                f"PANExtractor field '{field_name}' confidence "
                f"{field.confidence} is outside [0.0, 1.0] "
                f"with avg_confidence={confidence}"
            )

    @given(confidence=confidence_strategy)
    @settings(max_examples=100, deadline=2000)
    def test_driving_license_with_realistic_text_confidence_bounds(self, confidence: float) -> None:
        """DrivingLicenseExtractor with realistic text maintains confidence bounds."""
        text = "Driving License\nDL No: MH0220170012345\nName: RAJESH KUMAR\nDOB: 15/08/1990\nIssue Date: 01/01/2017\nValid Till: 01/01/2037\nAddress: 123 Road, Mumbai\nIssuing Authority: RTO Mumbai"
        ocr_result = _make_ocr_result(text, confidence)
        extractor = DrivingLicenseExtractor()

        result = extractor.extract(ocr_result)

        for field_name, field in result.items():
            assert 0.0 <= field.confidence <= 1.0, (
                f"DrivingLicenseExtractor field '{field_name}' confidence "
                f"{field.confidence} is outside [0.0, 1.0] "
                f"with avg_confidence={confidence}"
            )
