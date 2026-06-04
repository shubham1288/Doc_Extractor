"""Unit tests for ImagePreprocessor."""

import cv2
import numpy as np
import pytest

from app.services.preprocessor import ImagePreprocessor


@pytest.fixture
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor()


class TestToGrayscale:
    """Tests for ImagePreprocessor.to_grayscale method."""

    def test_color_image_converted_to_grayscale(self, preprocessor: ImagePreprocessor):
        """A 3-channel BGR image (100x100x3) is converted to (100x100)."""
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = preprocessor.to_grayscale(color_image)

        assert result.shape == (100, 100)
        assert result.ndim == 2

    def test_grayscale_image_returned_as_copy(self, preprocessor: ImagePreprocessor):
        """A grayscale image (100x100) is returned as a copy, not the same object."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = preprocessor.to_grayscale(gray_image)

        assert result.shape == (100, 100)
        assert result is not gray_image
        np.testing.assert_array_equal(result, gray_image)

    def test_output_dtype_is_uint8_for_color(self, preprocessor: ImagePreprocessor):
        """Output dtype is always uint8 for a color input."""
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = preprocessor.to_grayscale(color_image)

        assert result.dtype == np.uint8

    def test_output_dtype_is_uint8_for_grayscale(self, preprocessor: ImagePreprocessor):
        """Output dtype is always uint8 for a grayscale input."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = preprocessor.to_grayscale(gray_image)

        assert result.dtype == np.uint8

    def test_input_array_not_modified(self, preprocessor: ImagePreprocessor):
        """The original input array is never modified."""
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        original_copy = color_image.copy()

        preprocessor.to_grayscale(color_image)

        np.testing.assert_array_equal(color_image, original_copy)

    def test_bgra_image_converted_to_grayscale(self, preprocessor: ImagePreprocessor):
        """A 4-channel BGRA image is converted to single-channel grayscale."""
        bgra_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        result = preprocessor.to_grayscale(bgra_image)

        assert result.shape == (100, 100)
        assert result.ndim == 2
        assert result.dtype == np.uint8


class TestNormalizeResolution:
    """Tests for ImagePreprocessor.normalize_resolution method."""

    def test_upscale_from_72_to_300_dpi(self, preprocessor: ImagePreprocessor):
        """Image at 72 DPI is upscaled — dimensions increase by factor of ~4.17."""
        image = np.random.randint(0, 256, (100, 200), dtype=np.uint8)
        result = preprocessor.normalize_resolution(image, current_dpi=72, target_dpi=300)

        expected_scale = 300 / 72  # ~4.1667
        expected_height = int(round(100 * expected_scale))
        expected_width = int(round(200 * expected_scale))

        assert result.shape == (expected_height, expected_width)

    def test_same_dpi_returns_copy(self, preprocessor: ImagePreprocessor):
        """Image at 300 DPI with target 300 is returned unchanged (as a copy)."""
        image = np.random.randint(0, 256, (100, 200), dtype=np.uint8)
        result = preprocessor.normalize_resolution(image, current_dpi=300, target_dpi=300)

        assert result.shape == image.shape
        assert result is not image
        np.testing.assert_array_equal(result, image)

    def test_downscale_from_600_to_300_dpi(self, preprocessor: ImagePreprocessor):
        """Image at 600 DPI is downscaled — dimensions decrease by factor of 0.5."""
        image = np.random.randint(0, 256, (400, 800), dtype=np.uint8)
        result = preprocessor.normalize_resolution(image, current_dpi=600, target_dpi=300)

        expected_height = int(round(400 * 0.5))
        expected_width = int(round(800 * 0.5))

        assert result.shape == (expected_height, expected_width)

    def test_output_dtype_is_uint8(self, preprocessor: ImagePreprocessor):
        """Output dtype is always uint8 regardless of scaling direction."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        upscaled = preprocessor.normalize_resolution(image, current_dpi=72, target_dpi=300)
        assert upscaled.dtype == np.uint8

        downscaled = preprocessor.normalize_resolution(image, current_dpi=600, target_dpi=300)
        assert downscaled.dtype == np.uint8

        unchanged = preprocessor.normalize_resolution(image, current_dpi=300, target_dpi=300)
        assert unchanged.dtype == np.uint8


class TestAutoRotate:
    """Tests for ImagePreprocessor.auto_rotate method."""

    def test_correctly_oriented_image_no_rotation(self, preprocessor: ImagePreprocessor):
        """An image with predominantly horizontal lines is returned without rotation."""
        # Create an image with strong horizontal lines (simulating correctly oriented text)
        image = np.zeros((200, 400), dtype=np.uint8)
        # Draw horizontal lines to simulate text lines
        for y in range(20, 180, 20):
            cv2.line(image, (30, y), (370, y), 255, 2)

        result = preprocessor.auto_rotate(image)

        # Shape should remain the same (no 90-degree rotation)
        assert result.shape == image.shape

    def test_returns_new_array_not_input(self, preprocessor: ImagePreprocessor):
        """The auto_rotate method always returns a new array, never the input."""
        image = np.zeros((200, 400), dtype=np.uint8)
        # Draw horizontal lines
        for y in range(20, 180, 20):
            cv2.line(image, (30, y), (370, y), 255, 2)

        result = preprocessor.auto_rotate(image)

        assert result is not image

    def test_output_shape_correct_after_90_degree_rotation(self, preprocessor: ImagePreprocessor):
        """An image rotated 90 degrees (vertical lines dominate) produces correct output shape."""
        # Create an image with strong horizontal lines (correct orientation)
        original = np.zeros((200, 400), dtype=np.uint8)
        for y in range(20, 180, 20):
            cv2.line(original, (30, y), (370, y), 255, 2)

        # Rotate it 90 degrees clockwise to simulate a rotated input
        rotated_input = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
        # rotated_input is now (400, 200) with vertical lines

        result = preprocessor.auto_rotate(rotated_input)

        # After correction, the image should be rotated back
        # The corrected shape should be (height, width) where width > height
        # since the original was landscape (200, 400)
        assert result.shape[1] >= result.shape[0], (
            f"Expected landscape orientation after correction, got shape {result.shape}"
        )

    def test_input_not_modified(self, preprocessor: ImagePreprocessor):
        """The original input array is not modified by auto_rotate."""
        image = np.random.randint(0, 256, (200, 400), dtype=np.uint8)
        original_copy = image.copy()

        preprocessor.auto_rotate(image)

        np.testing.assert_array_equal(image, original_copy)

    def test_color_image_handled(self, preprocessor: ImagePreprocessor):
        """A 3-channel color image is processed without error."""
        color_image = np.zeros((200, 400, 3), dtype=np.uint8)
        # Draw horizontal lines in white on the color image
        for y in range(20, 180, 20):
            cv2.line(color_image, (30, y), (370, y), (255, 255, 255), 2)

        result = preprocessor.auto_rotate(color_image)

        # Should return without error, shape should match (no rotation needed)
        assert result.shape == color_image.shape


class TestDeskew:
    """Tests for ImagePreprocessor.deskew method."""

    def test_no_skew_returns_unchanged_copy(self, preprocessor: ImagePreprocessor):
        """An image with no skew (straight horizontal lines) returns an unchanged copy."""
        # Create an image with perfectly horizontal lines (no skew)
        image = np.ones((300, 400), dtype=np.uint8) * 255
        for y in range(30, 270, 30):
            cv2.line(image, (20, y), (380, y), 0, 2)

        result = preprocessor.deskew(image)

        # Image should be essentially unchanged since there's no skew
        assert result.shape == image.shape
        assert result is not image
        # The result should be very similar to the input (pixel-wise)
        np.testing.assert_array_equal(result, image)

    def test_significant_skew_is_corrected(self, preprocessor: ImagePreprocessor):
        """An image with significant skew (> 0.5 degrees) is corrected."""
        # Create a straight image with horizontal lines
        h, w = 300, 400
        straight_image = np.ones((h, w), dtype=np.uint8) * 255
        for y in range(30, 270, 30):
            cv2.line(straight_image, (20, y), (380, y), 0, 2)

        # Rotate the image by 5 degrees to simulate skew
        skew_angle = 5.0
        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        skewed_image = cv2.warpAffine(
            straight_image,
            rot_matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        result = preprocessor.deskew(skewed_image)

        # The result should NOT be identical to the skewed input
        # (it should have been corrected)
        assert not np.array_equal(result, skewed_image)
        assert result.shape == skewed_image.shape

    def test_input_not_modified(self, preprocessor: ImagePreprocessor):
        """The original input array is not modified by deskew."""
        # Create a skewed image
        h, w = 300, 400
        image = np.ones((h, w), dtype=np.uint8) * 255
        for y in range(30, 270, 30):
            cv2.line(image, (20, y), (380, y), 0, 2)
        # Apply a 3-degree skew
        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, 3.0, 1.0)
        skewed = cv2.warpAffine(
            image, rot_matrix, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )

        original_copy = skewed.copy()
        preprocessor.deskew(skewed)

        np.testing.assert_array_equal(skewed, original_copy)

    def test_output_dtype_is_uint8(self, preprocessor: ImagePreprocessor):
        """Output dtype is always uint8."""
        # Test with an image that has no detectable lines (returns copy)
        image = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
        result = preprocessor.deskew(image)
        assert result.dtype == np.uint8

        # Test with a skewed image that will be corrected
        h, w = 300, 400
        lined_image = np.ones((h, w), dtype=np.uint8) * 255
        for y in range(30, 270, 30):
            cv2.line(lined_image, (20, y), (380, y), 0, 2)
        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, 5.0, 1.0)
        skewed = cv2.warpAffine(
            lined_image, rot_matrix, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )
        result = preprocessor.deskew(skewed)
        assert result.dtype == np.uint8


class TestDenoise:
    """Tests for ImagePreprocessor.denoise method."""

    def test_output_shape_matches_input(self, preprocessor: ImagePreprocessor):
        """Output array has the same shape as the input."""
        image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        result = preprocessor.denoise(image)

        assert result.shape == image.shape

    def test_output_dtype_is_uint8(self, preprocessor: ImagePreprocessor):
        """Output dtype is uint8."""
        image = np.random.randint(0, 256, (80, 120), dtype=np.uint8)
        result = preprocessor.denoise(image)

        assert result.dtype == np.uint8

    def test_input_not_modified(self, preprocessor: ImagePreprocessor):
        """The original input array is not modified by denoise."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        original_copy = image.copy()

        preprocessor.denoise(image)

        np.testing.assert_array_equal(image, original_copy)


class TestEnhanceContrast:
    """Tests for ImagePreprocessor.enhance_contrast method."""

    def test_output_shape_matches_input(self, preprocessor: ImagePreprocessor):
        """Output array has the same shape as the input."""
        image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        result = preprocessor.enhance_contrast(image)

        assert result.shape == image.shape

    def test_output_dtype_is_uint8(self, preprocessor: ImagePreprocessor):
        """Output dtype is uint8."""
        image = np.random.randint(0, 256, (80, 120), dtype=np.uint8)
        result = preprocessor.enhance_contrast(image)

        assert result.dtype == np.uint8


class TestAdaptiveThreshold:
    """Tests for ImagePreprocessor.adaptive_threshold method."""

    def test_output_only_contains_0_and_255(self, preprocessor: ImagePreprocessor):
        """Output image has only binary values: 0 and 255."""
        image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        result = preprocessor.adaptive_threshold(image)

        unique_values = np.unique(result)
        assert all(v in (0, 255) for v in unique_values)

    def test_output_dtype_is_uint8(self, preprocessor: ImagePreprocessor):
        """Output dtype is uint8."""
        image = np.random.randint(0, 256, (80, 120), dtype=np.uint8)
        result = preprocessor.adaptive_threshold(image)

        assert result.dtype == np.uint8


class TestDetectBlur:
    """Tests for ImagePreprocessor.detect_blur method."""

    def test_returns_float(self, preprocessor: ImagePreprocessor):
        """detect_blur returns a float value."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = preprocessor.detect_blur(image)

        assert isinstance(result, float)

    def test_sharp_image_scores_higher_than_blurry(self, preprocessor: ImagePreprocessor):
        """A sharp image has a higher blur score than a heavily blurred image."""
        # Create a sharp image with edges
        sharp_image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(sharp_image, (50, 50), (150, 150), 255, 2)
        cv2.line(sharp_image, (0, 100), (200, 100), 255, 1)
        cv2.line(sharp_image, (100, 0), (100, 200), 255, 1)

        # Create a blurry version by applying a heavy Gaussian blur
        blurry_image = cv2.GaussianBlur(sharp_image, (31, 31), 10)

        sharp_score = preprocessor.detect_blur(sharp_image)
        blurry_score = preprocessor.detect_blur(blurry_image)

        assert sharp_score > blurry_score


class TestPreprocess:
    """Tests for ImagePreprocessor.preprocess pipeline method."""

    def test_preprocess_returns_grayscale_for_color_input(
        self, preprocessor: ImagePreprocessor
    ):
        """A color (3-channel) input produces a single-channel grayscale output."""
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = preprocessor.preprocess(color_image)

        assert result.ndim == 2, f"Expected 2D grayscale output, got ndim={result.ndim}"

    def test_preprocess_returns_new_array(self, preprocessor: ImagePreprocessor):
        """preprocess returns a new array, never the input object."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = preprocessor.preprocess(gray_image)

        assert result is not gray_image

    def test_preprocess_output_dtype_is_uint8(self, preprocessor: ImagePreprocessor):
        """The output dtype is always uint8."""
        color_image = np.random.randint(0, 256, (80, 120, 3), dtype=np.uint8)
        result = preprocessor.preprocess(color_image)

        assert result.dtype == np.uint8

    def test_preprocess_handles_various_image_sizes(
        self, preprocessor: ImagePreprocessor
    ):
        """preprocess works correctly for various image sizes."""
        sizes = [(50, 50), (100, 200), (200, 100), (150, 150)]
        for h, w in sizes:
            image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            result = preprocessor.preprocess(image)

            assert result.ndim == 2, f"Failed for size ({h}, {w})"
            assert result.dtype == np.uint8, f"Wrong dtype for size ({h}, {w})"
            # Resolution normalization (72 → 300 DPI) scales by ~4.17x
            # The output should be larger than the input
            assert result.shape[0] > h, f"Height not scaled for size ({h}, {w})"
            assert result.shape[1] > w, f"Width not scaled for size ({h}, {w})"

    def test_preprocess_does_not_modify_input(self, preprocessor: ImagePreprocessor):
        """The original input array is never modified."""
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        original_copy = color_image.copy()

        preprocessor.preprocess(color_image)

        np.testing.assert_array_equal(color_image, original_copy)

    def test_preprocess_grayscale_input_produces_grayscale_output(
        self, preprocessor: ImagePreprocessor
    ):
        """A grayscale input also produces a single-channel grayscale output."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = preprocessor.preprocess(gray_image)

        assert result.ndim == 2
        assert result.dtype == np.uint8
