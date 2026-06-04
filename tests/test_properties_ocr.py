"""
Property-based tests for OCR Engine — singleton identity, confidence aggregation,
and batch result count.

Uses Hypothesis to verify that OCR engine properties hold for ALL generated inputs.

**Validates: Requirements 3.1, 3.2, 3.4**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from unittest.mock import patch

import numpy as np

from app.models.ocr import OCRResult
from app.ocr.engine import OCREngine


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset OCREngine singleton before and after each test."""
    OCREngine.reset()
    yield
    OCREngine.reset()


# ---------------------------------------------------------------------------
# Property 8: OCR engine singleton identity
# ---------------------------------------------------------------------------


@patch("app.ocr.engine._PADDLE_AVAILABLE", False)
class TestSingletonIdentityProperty:
    """Property 8: OCR engine singleton identity.

    For ANY number of calls to the OCREngine constructor, all calls SHALL
    return the exact same object instance.

    **Validates: Requirements 3.1**
    """

    @given(num_calls=st.integers(min_value=2, max_value=20))
    @settings(max_examples=100, deadline=2000)
    def test_all_instantiations_return_same_object(self, num_calls: int) -> None:
        """For any number of OCREngine() calls (2-20), all returned instances are identical."""
        # Reset before each hypothesis example to ensure clean state
        OCREngine.reset()

        instances = [OCREngine() for _ in range(num_calls)]

        # All instances must be the same object (identity check with `is`)
        first = instances[0]
        for i, inst in enumerate(instances[1:], start=1):
            assert inst is first, (
                f"Instance #{i} is not the same object as instance #0. "
                f"id(first)={id(first)}, id(instances[{i}])={id(inst)}"
            )

    @given(num_calls=st.integers(min_value=2, max_value=20))
    @settings(max_examples=100, deadline=2000)
    def test_singleton_id_is_constant(self, num_calls: int) -> None:
        """For any number of calls, id() of all instances is the same."""
        OCREngine.reset()

        ids = [id(OCREngine()) for _ in range(num_calls)]

        assert len(set(ids)) == 1, (
            f"Expected all {num_calls} instances to have same id, "
            f"but got {len(set(ids))} distinct ids: {set(ids)}"
        )


# ---------------------------------------------------------------------------
# Property 9: Confidence aggregation consistency
# ---------------------------------------------------------------------------


@patch("app.ocr.engine._PADDLE_AVAILABLE", False)
class TestConfidenceAggregationProperty:
    """Property 9: OCR confidence aggregation consistency.

    For ANY list of confidence values (floats between 0.0 and 1.0), the
    avg_confidence in an OCRResult must equal the arithmetic mean of the
    confidences list.

    **Validates: Requirements 3.2**
    """

    @given(
        confidences=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=200, deadline=2000)
    def test_avg_confidence_equals_arithmetic_mean(self, confidences: list[float]) -> None:
        """For any list of confidence values, _parse_result produces correct avg_confidence."""
        # Construct a mock PaddleOCR output format:
        # [ [ [box, (text, confidence)], ... ] ]
        raw_result = [[
            [
                [[0, 0], [100, 0], [100, 30], [0, 30]],
                (f"text_{i}", conf),
            ]
            for i, conf in enumerate(confidences)
        ]]

        result = OCREngine._parse_result(raw_result)

        expected_avg = sum(confidences) / len(confidences)
        assert result.avg_confidence == pytest.approx(expected_avg, rel=1e-9), (
            f"Expected avg_confidence={expected_avg}, got {result.avg_confidence}. "
            f"Confidences: {confidences}"
        )

    @given(
        confidences=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=200, deadline=2000)
    def test_confidences_list_preserved_in_result(self, confidences: list[float]) -> None:
        """The confidences list in the parsed result matches the input confidences."""
        raw_result = [[
            [
                [[0, 0], [100, 0], [100, 30], [0, 30]],
                (f"word_{i}", conf),
            ]
            for i, conf in enumerate(confidences)
        ]]

        result = OCREngine._parse_result(raw_result)

        assert len(result.confidences) == len(confidences)
        for actual, expected in zip(result.confidences, confidences):
            assert actual == pytest.approx(expected, rel=1e-9)

    def test_empty_result_has_zero_avg_confidence(self) -> None:
        """An empty parse result produces avg_confidence of 0.0."""
        result = OCREngine._parse_result(None)
        assert result.avg_confidence == 0.0

        result = OCREngine._parse_result([])
        assert result.avg_confidence == 0.0

        result = OCREngine._parse_result([[]])
        assert result.avg_confidence == 0.0


# ---------------------------------------------------------------------------
# Property 10: Batch OCR result count
# ---------------------------------------------------------------------------


@patch("app.ocr.engine._PADDLE_AVAILABLE", False)
class TestBatchResultCountProperty:
    """Property 10: Batch OCR result count.

    For ANY list of N input images provided to batch processing, the OCR
    engine SHALL return exactly N OCRResult objects.

    **Validates: Requirements 3.4**
    """

    @given(num_images=st.integers(min_value=0, max_value=20))
    @settings(max_examples=100, deadline=5000)
    def test_batch_returns_exactly_n_results(self, num_images: int) -> None:
        """For any number of images (0-20), extract_batch returns exactly that many results."""
        OCREngine.reset()
        engine = OCREngine()

        # Generate random images of varying sizes
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8)
            for _ in range(num_images)
        ]

        results = engine.extract_batch(images)

        assert len(results) == num_images, (
            f"Expected {num_images} results, got {len(results)}"
        )

    @given(num_images=st.integers(min_value=0, max_value=20))
    @settings(max_examples=100, deadline=5000)
    def test_batch_results_are_all_ocr_result_instances(self, num_images: int) -> None:
        """Every element in batch result is an OCRResult instance."""
        OCREngine.reset()
        engine = OCREngine()

        images = [
            np.zeros((50, 50, 3), dtype=np.uint8)
            for _ in range(num_images)
        ]

        results = engine.extract_batch(images)

        assert all(isinstance(r, OCRResult) for r in results), (
            f"Not all results are OCRResult instances. "
            f"Types: {[type(r).__name__ for r in results]}"
        )

    @given(
        num_images=st.integers(min_value=1, max_value=15),
        height=st.integers(min_value=10, max_value=500),
        width=st.integers(min_value=10, max_value=500),
    )
    @settings(max_examples=50, deadline=5000)
    def test_batch_count_independent_of_image_dimensions(
        self, num_images: int, height: int, width: int
    ) -> None:
        """Result count equals input count regardless of image dimensions."""
        OCREngine.reset()
        engine = OCREngine()

        images = [
            np.zeros((height, width, 3), dtype=np.uint8)
            for _ in range(num_images)
        ]

        results = engine.extract_batch(images)

        assert len(results) == num_images, (
            f"Expected {num_images} results for {height}x{width} images, "
            f"got {len(results)}"
        )
