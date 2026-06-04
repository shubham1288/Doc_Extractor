"""
Property-based tests for Image Preprocessor — immutability, grayscale conversion,
and preprocessing idempotence.

Uses Hypothesis to verify that the following properties hold for ALL generated inputs:

- Property 5: Preprocessing immutability — input array is never modified
- Property 6: Grayscale conversion — color images always produce 2D uint8 output
- Property 7: Preprocessing idempotence — output is always 2D uint8

**Validates: Requirements 2.1, 2.2, 2.3, 2.5, 2.6**
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from app.services.preprocessor import ImagePreprocessor


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Image dimension strategies (keep small for speed)
_width_strategy = st.integers(min_value=20, max_value=200)
_height_strategy = st.integers(min_value=20, max_value=200)


# Strategy for generating random grayscale images (2D, uint8)
@st.composite
def grayscale_images(draw):
    """Generate random 2D grayscale images."""
    h = draw(_height_strategy)
    w = draw(_width_strategy)
    return draw(arrays(np.uint8, shape=(h, w)))


# Strategy for generating random color images (3-channel, uint8)
@st.composite
def color_images(draw):
    """Generate random 3-channel BGR color images."""
    h = draw(_height_strategy)
    w = draw(_width_strategy)
    return draw(arrays(np.uint8, shape=(h, w, 3)))


# Strategy for generating any valid image (grayscale or color)
@st.composite
def any_images(draw):
    """Generate either a grayscale or color image."""
    is_color = draw(st.booleans())
    if is_color:
        return draw(color_images())
    else:
        return draw(grayscale_images())


# ---------------------------------------------------------------------------
# Property 5: Preprocessing immutability
# ---------------------------------------------------------------------------


class TestPreprocessingImmutabilityProperty:
    """Property 5: Preprocessing immutability.

    For ANY input image (grayscale or color), calling preprocess() must NEVER
    modify the input array. The input array must be byte-for-byte identical
    before and after calling preprocess().

    **Validates: Requirements 2.6**
    """

    @given(image=grayscale_images())
    @settings(max_examples=50, deadline=5000)
    def test_preprocess_does_not_modify_grayscale_input(
        self, image: np.ndarray
    ) -> None:
        """For any grayscale image, preprocess() leaves the input unchanged."""
        preprocessor = ImagePreprocessor()

        # Save a copy of the original for comparison
        original = image.copy()

        # Call preprocess — this must not modify the input
        _ = preprocessor.preprocess(image)

        # Verify byte-for-byte identical
        np.testing.assert_array_equal(
            image,
            original,
            err_msg="preprocess() modified the input grayscale array",
        )

    @given(image=color_images())
    @settings(max_examples=50, deadline=5000)
    def test_preprocess_does_not_modify_color_input(
        self, image: np.ndarray
    ) -> None:
        """For any color image, preprocess() leaves the input unchanged."""
        preprocessor = ImagePreprocessor()

        # Save a copy of the original for comparison
        original = image.copy()

        # Call preprocess — this must not modify the input
        _ = preprocessor.preprocess(image)

        # Verify byte-for-byte identical
        np.testing.assert_array_equal(
            image,
            original,
            err_msg="preprocess() modified the input color array",
        )

    @given(image=any_images())
    @settings(max_examples=50, deadline=5000)
    def test_preprocess_does_not_modify_any_input(
        self, image: np.ndarray
    ) -> None:
        """For any image (grayscale or color), preprocess() leaves the input unchanged."""
        preprocessor = ImagePreprocessor()

        # Save a copy of the original for comparison
        original = image.copy()

        # Call preprocess — this must not modify the input
        _ = preprocessor.preprocess(image)

        # Verify byte-for-byte identical
        np.testing.assert_array_equal(
            image,
            original,
            err_msg="preprocess() modified the input array",
        )


# ---------------------------------------------------------------------------
# Property 6: Grayscale conversion
# ---------------------------------------------------------------------------


class TestGrayscaleConversionProperty:
    """Property 6: Grayscale conversion.

    For ANY 3-channel color image, to_grayscale() must always produce a 2D
    single-channel output with dtype uint8.

    For ANY grayscale image (2D), to_grayscale() must return a copy with
    the same values and dtype uint8.

    **Validates: Requirements 2.3**
    """

    @given(image=color_images())
    @settings(max_examples=50, deadline=5000)
    def test_color_image_produces_2d_output(
        self, image: np.ndarray
    ) -> None:
        """For any color image, to_grayscale() produces a 2D single-channel output."""
        preprocessor = ImagePreprocessor()

        result = preprocessor.to_grayscale(image)

        # Must be 2D (single channel)
        assert result.ndim == 2, (
            f"Expected 2D output, got ndim={result.ndim}, shape={result.shape}"
        )
        # Must preserve spatial dimensions
        assert result.shape[0] == image.shape[0], "Height mismatch"
        assert result.shape[1] == image.shape[1], "Width mismatch"

    @given(image=color_images())
    @settings(max_examples=50, deadline=5000)
    def test_color_image_output_is_uint8(
        self, image: np.ndarray
    ) -> None:
        """For any color image, to_grayscale() output dtype is always uint8."""
        preprocessor = ImagePreprocessor()

        result = preprocessor.to_grayscale(image)

        assert result.dtype == np.uint8, (
            f"Expected dtype uint8, got {result.dtype}"
        )

    @given(image=grayscale_images())
    @settings(max_examples=50, deadline=5000)
    def test_grayscale_image_returns_copy_with_same_values(
        self, image: np.ndarray
    ) -> None:
        """For any grayscale image, to_grayscale() returns a copy with the same values."""
        preprocessor = ImagePreprocessor()

        result = preprocessor.to_grayscale(image)

        # Must be 2D
        assert result.ndim == 2, (
            f"Expected 2D output for grayscale input, got ndim={result.ndim}"
        )
        # Must have same values
        np.testing.assert_array_equal(
            result,
            image,
            err_msg="to_grayscale() did not preserve values for grayscale input",
        )
        # Must be a copy (not the same object)
        assert result is not image, "to_grayscale() returned the same object, not a copy"

    @given(image=grayscale_images())
    @settings(max_examples=50, deadline=5000)
    def test_grayscale_image_output_is_uint8(
        self, image: np.ndarray
    ) -> None:
        """For any grayscale image, to_grayscale() output dtype is always uint8."""
        preprocessor = ImagePreprocessor()

        result = preprocessor.to_grayscale(image)

        assert result.dtype == np.uint8, (
            f"Expected dtype uint8, got {result.dtype}"
        )

    @given(image=any_images())
    @settings(max_examples=50, deadline=5000)
    def test_output_always_2d_and_uint8(
        self, image: np.ndarray
    ) -> None:
        """For any valid image input, to_grayscale() always produces 2D uint8."""
        preprocessor = ImagePreprocessor()

        result = preprocessor.to_grayscale(image)

        assert result.ndim == 2, (
            f"Expected 2D output, got ndim={result.ndim}, input shape={image.shape}"
        )
        assert result.dtype == np.uint8, (
            f"Expected dtype uint8, got {result.dtype}"
        )


# ---------------------------------------------------------------------------
# Property 7: Preprocessing idempotence (approximate)
# ---------------------------------------------------------------------------


class TestPreprocessingIdempotenceProperty:
    """Property 7: Preprocessing idempotence (approximate).

    For ANY image, applying preprocess():
    - Output is always 2D (grayscale)
    - Output is always uint8
    - Output dimensions are proportional to input (within expected scaling)

    Due to resolution normalization (72->300 DPI assumption), true idempotence
    is not guaranteed. Instead we verify structural properties of the output.

    **Validates: Requirements 2.1, 2.2, 2.5**
    """

    @given(image=any_images())
    @settings(max_examples=50, deadline=5000)
    def test_preprocess_output_is_always_2d(
        self, image: np.ndarray
    ) -> None:
        """For any image, preprocess() output is always 2D (grayscale)."""
        preprocessor = ImagePreprocessor()

        result = preprocessor.preprocess(image)

        assert result.ndim == 2, (
            f"Expected 2D output, got ndim={result.ndim}, shape={result.shape}"
        )

    @given(image=any_images())
    @settings(max_examples=50, deadline=5000)
    def test_preprocess_output_is_always_uint8(
        self, image: np.ndarray
    ) -> None:
        """For any image, preprocess() output dtype is always uint8."""
        preprocessor = ImagePreprocessor()

        result = preprocessor.preprocess(image)

        assert result.dtype == np.uint8, (
            f"Expected dtype uint8, got {result.dtype}"
        )

    @given(image=any_images())
    @settings(max_examples=50, deadline=5000)
    def test_preprocess_output_dimensions_proportional_to_input(
        self, image: np.ndarray
    ) -> None:
        """For any image, preprocess() output dimensions reflect the expected DPI scaling.

        The preprocessor scales from assumed 72 DPI to 300 DPI (factor ~4.17).
        Auto-rotation may swap width/height, so we check that the output area
        is approximately proportional to the input area scaled by the factor squared.
        """
        preprocessor = ImagePreprocessor()

        result = preprocessor.preprocess(image)

        # Input spatial dimensions
        input_h, input_w = image.shape[0], image.shape[1]
        input_area = input_h * input_w

        # Output spatial dimensions
        output_h, output_w = result.shape[0], result.shape[1]
        output_area = output_h * output_w

        # Expected scale factor from 72 to 300 DPI
        scale_factor = 300 / 72  # ~4.17
        expected_area = input_area * (scale_factor ** 2)

        # Allow generous tolerance (auto-rotation, deskew, rounding can change dimensions)
        # The output area should be within 50% of the expected scaled area
        assert output_area > expected_area * 0.5, (
            f"Output area {output_area} is too small. "
            f"Expected ~{expected_area:.0f} (input area {input_area} * scale^2)"
        )
        assert output_area < expected_area * 1.5, (
            f"Output area {output_area} is too large. "
            f"Expected ~{expected_area:.0f} (input area {input_area} * scale^2)"
        )

    @given(image=any_images())
    @settings(max_examples=30, deadline=30000)
    def test_preprocess_twice_produces_same_shape_type(
        self, image: np.ndarray
    ) -> None:
        """Applying preprocess() twice produces output with same dtype and ndim as single application."""
        preprocessor = ImagePreprocessor()

        result_once = preprocessor.preprocess(image)
        result_twice = preprocessor.preprocess(result_once)

        # Both applications produce 2D uint8 output
        assert result_twice.ndim == 2, (
            f"Second preprocess() produced ndim={result_twice.ndim}, expected 2"
        )
        assert result_twice.dtype == np.uint8, (
            f"Second preprocess() produced dtype={result_twice.dtype}, expected uint8"
        )
