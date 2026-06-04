"""Image preprocessing pipeline for KYC document OCR enhancement."""

import cv2
import numpy as np
from numpy.typing import NDArray


class ImagePreprocessor:
    """Pipeline for image enhancement before OCR."""

    def to_grayscale(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert an image to grayscale.

        Args:
            image: Input image as a numpy array (2D grayscale, 3-channel BGR,
                   or 4-channel BGRA).

        Returns:
            A new single-channel grayscale image array. The input is never modified.
        """
        if image.ndim == 2:
            # Already grayscale — return a copy to avoid mutating the input
            return image.copy()

        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        raise ValueError(
            f"Unsupported image format: ndim={image.ndim}, shape={image.shape}"
        )

    def auto_rotate(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Detect and correct image rotation (0, 90, 180, or 270 degrees).

        Uses edge orientation analysis to determine if the image is rotated.
        Applies Canny edge detection followed by Hough line analysis to count
        horizontal vs vertical line segments. If vertical lines dominate,
        the image is likely rotated 90 or 270 degrees.

        Args:
            image: Input image as a numpy array (uint8, grayscale or color).

        Returns:
            A new image array corrected for rotation. If no rotation is needed,
            returns a copy of the input. The input array is never modified.
        """
        # Work on a grayscale version for edge analysis
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use probabilistic Hough Line Transform to detect line segments
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=40,
            maxLineGap=10,
        )

        if lines is None:
            # No lines detected — cannot determine orientation, return copy
            return image.copy()

        horizontal_count = 0
        vertical_count = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # Classify line as horizontal or vertical based on angle
            # A line is horizontal if dx >> dy (angle close to 0 degrees)
            # A line is vertical if dy >> dx (angle close to 90 degrees)
            if dx == 0 and dy == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 30:
                horizontal_count += 1
            elif angle > 60:
                vertical_count += 1

        # Determine rotation based on edge distribution
        # For KYC documents, text creates predominantly horizontal edges
        # when correctly oriented. If vertical edges dominate, image is
        # likely rotated by 90 degrees.
        total_lines = horizontal_count + vertical_count
        if total_lines == 0:
            return image.copy()

        vertical_ratio = vertical_count / total_lines

        if vertical_ratio > 0.65:
            # More vertical lines than horizontal — image is likely rotated 90°
            # Rotate 90° counterclockwise to correct
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # If horizontal lines dominate or it's roughly balanced,
        # image is likely already correctly oriented
        return image.copy()

    def deskew(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Correct skew angle using Hough Line Transform.

        Detects the skew angle of the image by analyzing line orientations.
        If the absolute skew angle exceeds 0.5 degrees, the image is rotated
        to correct the skew. Otherwise, a copy of the image is returned unchanged.

        Args:
            image: Input image as a numpy array (uint8, grayscale or color).

        Returns:
            A new image array with skew corrected. If the skew is 0.5 degrees
            or less, returns a copy of the input. The input array is never modified.
        """
        # Work on a grayscale version for edge analysis
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use standard Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            # No lines detected — cannot determine skew, return copy
            return image.copy()

        # Calculate angles from detected lines, filtering to near-horizontal range
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert theta to degrees relative to horizontal
            # theta is in radians from 0 to pi
            # Horizontal lines have theta near pi/2 (90 degrees)
            angle_deg = np.degrees(theta) - 90.0
            # Filter to near-horizontal range (-45 to 45 degrees)
            if -45.0 <= angle_deg <= 45.0:
                angles.append(angle_deg)

        if not angles:
            # No near-horizontal lines found
            return image.copy()

        # Use median angle as the skew estimate (robust to outliers)
        median_angle = float(np.median(angles))

        # Only correct if skew exceeds 0.5 degrees
        if abs(median_angle) <= 0.5:
            return image.copy()

        # Rotate the image to correct the skew
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return corrected

    def denoise(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Remove noise using bilateral filtering.

        Applies a bilateral filter that smooths flat regions while preserving
        edges, making it ideal for document images with text.

        Args:
            image: Input image as a numpy array (uint8, grayscale or color).

        Returns:
            A new denoised image array. The input is never modified.
        """
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def enhance_contrast(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Enhances local contrast which helps OCR accuracy on unevenly lit
        documents. The image must be single-channel grayscale.

        Args:
            image: Input grayscale image as a numpy array (uint8, 2D).

        Returns:
            A new contrast-enhanced image array. The input is never modified.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def adaptive_threshold(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply adaptive thresholding for binarization.

        Converts a grayscale image to binary using Gaussian-weighted
        adaptive thresholding. Useful for low-contrast documents where
        global thresholding would fail.

        Args:
            image: Input grayscale image as a numpy array (uint8, 2D).

        Returns:
            A new binary image array with pixel values of 0 or 255.
            The input is never modified.
        """
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def detect_blur(self, image: NDArray[np.uint8]) -> float:
        """Return blur score using Laplacian variance.

        Computes the variance of the Laplacian of the image as a measure
        of image sharpness. Higher values indicate a sharper image, lower
        values indicate more blur.

        A threshold of 100 is typically used: values below 100 suggest the
        image is blurry and may benefit from denoising rather than sharpening.

        Args:
            image: Input image as a numpy array (uint8, grayscale or color).

        Returns:
            The Laplacian variance as a float. Higher = sharper, lower = blurrier.
        """
        return float(cv2.Laplacian(image, cv2.CV_64F).var())

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Run full preprocessing pipeline for OCR enhancement.

        Orchestrates all preprocessing steps in order:
        1. Convert to grayscale (if color)
        2. Normalize resolution to 300 DPI (assume input is 72 DPI)
        3. Auto-rotate (0/90/180/270 degree correction)
        4. Deskew using Hough transform (only if skew > 0.5 degrees)
        5. Denoise using bilateral filter (only if blur_score < 100)
        6. Enhance contrast using CLAHE
        7. Apply adaptive thresholding (only if std < 40)

        Args:
            image: Input image as a numpy array (uint8, grayscale or color).

        Returns:
            A new single-channel uint8 grayscale image. The input is never modified.
        """
        # Step 1: Convert to grayscale
        result = self.to_grayscale(image)

        # Step 2: Normalize resolution to 300 DPI (assume input is 72 DPI)
        result = self.normalize_resolution(result, current_dpi=72, target_dpi=300)

        # Step 3: Auto-rotate
        result = self.auto_rotate(result)

        # Step 4: Deskew (conditional — deskew internally checks > 0.5 degrees)
        result = self.deskew(result)

        # Step 5: Denoise only if image is blurry (blur_score < 100)
        blur_score = self.detect_blur(result)
        if blur_score < 100:
            result = self.denoise(result)

        # Step 6: Enhance contrast using CLAHE
        result = self.enhance_contrast(result)

        # Step 7: Adaptive thresholding only if low contrast (std < 40)
        if float(np.std(result)) < 40:
            result = self.adaptive_threshold(result)

        return result

    def normalize_resolution(
        self,
        image: NDArray[np.uint8],
        current_dpi: int = 72,
        target_dpi: int = 300,
    ) -> NDArray[np.uint8]:
        """Scale image to target DPI for consistent OCR.

        Args:
            image: Input image as a numpy array (uint8).
            current_dpi: The current resolution of the image in DPI.
            target_dpi: The desired resolution in DPI (default 300).

        Returns:
            A new image array scaled to the target DPI. If the scaling factor
            is approximately 1.0 (within 0.01), a copy of the original is returned.
        """
        scale_factor = target_dpi / current_dpi

        if abs(scale_factor - 1.0) < 0.01:
            return image.copy()

        new_width = int(round(image.shape[1] * scale_factor))
        new_height = int(round(image.shape[0] * scale_factor))

        interpolation = (
            cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA
        )

        resized = cv2.resize(
            image, (new_width, new_height), interpolation=interpolation
        )
        return resized
