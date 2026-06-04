"""
Property-based tests for PII redaction in log entries.

Verifies that for ANY log output that passes through the PII filter,
no Aadhaar or PAN number patterns remain in the output.

**Validates: Requirements 12.2, 12.3**
"""

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from app.middleware.pii_filter import (
    AADHAAR_PATTERN,
    PAN_PATTERN,
    REDACTED_AADHAAR,
    REDACTED_PAN,
    redact_pii,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: Aadhaar numbers (4 digits, optional space, 4 digits, optional space, 4 digits)
aadhaar_strategy = st.builds(
    lambda d1, s1, d2, s2, d3: f"{d1}{s1}{d2}{s2}{d3}",
    d1=st.from_regex(r"\d{4}", fullmatch=True),
    s1=st.sampled_from(["", " "]),
    d2=st.from_regex(r"\d{4}", fullmatch=True),
    s2=st.sampled_from(["", " "]),
    d3=st.from_regex(r"\d{4}", fullmatch=True),
)

# Strategy: PAN numbers (5 uppercase letters, 4 digits, 1 uppercase letter)
pan_strategy = st.from_regex(r"[A-Z]{5}\d{4}[A-Z]", fullmatch=True)

# Strategy: arbitrary log event text that may contain PII
arbitrary_text_strategy = st.text(min_size=0, max_size=500)

# Strategy: text that contains an embedded Aadhaar number
text_with_aadhaar_strategy = st.builds(
    lambda prefix, aadhaar, suffix: f"{prefix}{aadhaar}{suffix}",
    prefix=st.text(min_size=0, max_size=50, alphabet=st.characters(whitelist_categories=("L", "Zs"))),
    aadhaar=aadhaar_strategy,
    suffix=st.text(min_size=0, max_size=50, alphabet=st.characters(whitelist_categories=("L", "Zs"))),
)

# Strategy: text that contains an embedded PAN number
text_with_pan_strategy = st.builds(
    lambda prefix, pan, suffix: f"{prefix}{pan}{suffix}",
    prefix=st.text(min_size=0, max_size=50, alphabet=st.characters(whitelist_categories=("L", "Zs"))),
    pan=pan_strategy,
    suffix=st.text(min_size=0, max_size=50, alphabet=st.characters(whitelist_categories=("L", "Zs"))),
)

# Strategy: log event dict keys (realistic field names)
log_key_strategy = st.sampled_from([
    "event", "message", "data", "user_info", "detail", "description",
    "note", "raw_text", "ocr_text", "extracted", "value", "field",
])

# Strategy: log event dicts with PII embedded in values
event_dict_with_aadhaar_strategy = st.fixed_dictionaries({
    "event": st.just("document_processed"),
}).map(lambda d: d)  # base dict


# ---------------------------------------------------------------------------
# Property 20: No PII in logs
# ---------------------------------------------------------------------------


class TestNoPIIInLogsProperty:
    """Property 20: No PII in logs.

    For ANY log entry produced during request processing that passes through
    the PII filter, it SHALL NOT contain any Aadhaar or PAN patterns.

    **Validates: Requirements 12.2, 12.3**
    """

    @given(aadhaar=aadhaar_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_aadhaar_pattern_always_redacted_in_event(self, aadhaar: str) -> None:
        """For any Aadhaar number in the event field, it is redacted after filtering."""
        event_dict = {"event": f"Processing document with number {aadhaar}"}
        result = redact_pii(None, "info", event_dict)

        assert not AADHAAR_PATTERN.search(result["event"]), (
            f"Aadhaar pattern still present after redaction: {result['event']}"
        )
        assert REDACTED_AADHAAR in result["event"]

    @given(pan=pan_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_pan_pattern_always_redacted_in_event(self, pan: str) -> None:
        """For any PAN number in the event field, it is redacted after filtering."""
        event_dict = {"event": f"Found PAN: {pan}"}
        result = redact_pii(None, "info", event_dict)

        assert not PAN_PATTERN.search(result["event"]), (
            f"PAN pattern still present after redaction: {result['event']}"
        )
        assert REDACTED_PAN in result["event"]

    @given(
        aadhaar=aadhaar_strategy,
        key=log_key_strategy,
    )
    @settings(max_examples=500, deadline=2000)
    def test_aadhaar_redacted_in_any_value_field(self, aadhaar: str, key: str) -> None:
        """For any Aadhaar number in any log value field, it is redacted."""
        event_dict = {"event": "test_event", key: f"data contains {aadhaar} here"}
        result = redact_pii(None, "info", event_dict)

        assert not AADHAAR_PATTERN.search(result[key]), (
            f"Aadhaar pattern still present in field '{key}': {result[key]}"
        )

    @given(
        pan=pan_strategy,
        key=log_key_strategy,
    )
    @settings(max_examples=500, deadline=2000)
    def test_pan_redacted_in_any_value_field(self, pan: str, key: str) -> None:
        """For any PAN number in any log value field, it is redacted."""
        event_dict = {"event": "test_event", key: f"data contains {pan} here"}
        result = redact_pii(None, "info", event_dict)

        assert not PAN_PATTERN.search(result[key]), (
            f"PAN pattern still present in field '{key}': {result[key]}"
        )

    @given(text=text_with_aadhaar_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_aadhaar_in_arbitrary_surrounding_text_is_redacted(self, text: str) -> None:
        """For any text containing an Aadhaar pattern with arbitrary surrounding text, it is redacted."""
        event_dict = {"event": text, "message": text}
        result = redact_pii(None, "info", event_dict)

        for key in ("event", "message"):
            assert not AADHAAR_PATTERN.search(result[key]), (
                f"Aadhaar pattern still present in field '{key}': {result[key]}"
            )

    @given(text=text_with_pan_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_pan_in_arbitrary_surrounding_text_is_redacted(self, text: str) -> None:
        """For any text containing a PAN pattern with arbitrary surrounding text, it is redacted."""
        event_dict = {"event": text, "data": text}
        result = redact_pii(None, "info", event_dict)

        for key in ("event", "data"):
            assert not PAN_PATTERN.search(result[key]), (
                f"PAN pattern still present in field '{key}': {result[key]}"
            )

    @given(
        aadhaar=aadhaar_strategy,
        pan=pan_strategy,
    )
    @settings(max_examples=300, deadline=2000)
    def test_both_aadhaar_and_pan_redacted_simultaneously(self, aadhaar: str, pan: str) -> None:
        """For any log entry containing both Aadhaar and PAN, both are redacted."""
        combined_text = f"Aadhaar: {aadhaar}, PAN: {pan}"
        event_dict = {"event": combined_text, "detail": combined_text}
        result = redact_pii(None, "info", event_dict)

        for key in ("event", "detail"):
            assert not AADHAAR_PATTERN.search(result[key]), (
                f"Aadhaar still in '{key}': {result[key]}"
            )
            assert not PAN_PATTERN.search(result[key]), (
                f"PAN still in '{key}': {result[key]}"
            )

    @given(text=arbitrary_text_strategy)
    @settings(max_examples=500, deadline=2000)
    def test_non_pii_text_passes_through_unchanged(self, text: str) -> None:
        """For any text without PII patterns, the filter does not alter it."""
        # Remove any accidental PII patterns from the generated text
        clean_text = AADHAAR_PATTERN.sub("XXXX", text)
        clean_text = PAN_PATTERN.sub("YYYYY", clean_text)

        event_dict = {"event": clean_text}
        result = redact_pii(None, "info", event_dict)

        assert result["event"] == clean_text, (
            f"Non-PII text was modified: input='{clean_text}', output='{result['event']}'"
        )

    @given(value=st.integers())
    @settings(max_examples=200, deadline=2000)
    def test_non_string_values_pass_through_unchanged(self, value: int) -> None:
        """For any non-string value in log entries, it passes through unchanged."""
        event_dict = {"event": "test", "count": value}
        result = redact_pii(None, "info", event_dict)

        assert result["count"] == value
