"""
Tests for PDF processing utilities.

Tests PDF-to-image conversion logic, page limit enforcement,
and multi-page OCR result merging. Mocks pdf2image to avoid
requiring poppler-utils system dependency.

Includes property-based tests for page limit enforcement (Property 19).
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.models.ocr import OCRResult
from app.services.pdf_processor import pdf_to_images, merge_ocr_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 100, height: int = 80) -> Image.Image:
    """Create a simple RGB PIL image for mock returns."""
    return Image.fromarray(
        np.random.randint(0, 256, (height, width, 3), dtype=np.uint8), mode="RGB"
    )


def _make_ocr_result(
    text: str = "sample text",
    boxes: list | None = None,
    confidences: list[float] | None = None,
) -> OCRResult:
    """Helper to create OCRResult instances for testing."""
    if boxes is None:
        boxes = [[[0, 0], [100, 0], [100, 20], [0, 20]]]
    if confidences is None:
        confidences = [0.9]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(
        text=text,
        boxes=boxes,
        confidences=confidences,
        avg_confidence=avg_conf,
    )


# ---------------------------------------------------------------------------
# Unit Tests: pdf_to_images
# ---------------------------------------------------------------------------


class TestPdfToImages:
    """Unit tests for pdf_to_images function."""

    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    @patch("app.services.pdf_processor.convert_from_bytes")
    def test_converts_single_page_pdf(self, mock_convert, mock_pdfinfo):
        """Single-page PDF returns one BGR numpy array."""
        mock_pdfinfo.return_value = {"Pages": 1}
        pil_img = _make_pil_image(200, 150)
        mock_convert.return_value = [pil_img]

        result = pdf_to_images(b"fake-pdf-bytes", dpi=300)

        assert len(result) == 1
        assert result[0].shape == (150, 200, 3)
        assert result[0].dtype == np.uint8
        mock_convert.assert_called_once_with(b"fake-pdf-bytes", dpi=300)

    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    @patch("app.services.pdf_processor.convert_from_bytes")
    def test_converts_multi_page_pdf(self, mock_convert, mock_pdfinfo):
        """Multi-page PDF returns multiple BGR numpy arrays."""
        mock_pdfinfo.return_value = {"Pages": 3}
        pages = [_make_pil_image() for _ in range(3)]
        mock_convert.return_value = pages

        result = pdf_to_images(b"fake-pdf-bytes", dpi=300)

        assert len(result) == 3
        for img in result:
            assert img.ndim == 3
            assert img.shape[2] == 3

    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    @patch("app.services.pdf_processor.convert_from_bytes")
    def test_enforces_page_limit(self, mock_convert, mock_pdfinfo):
        """PDF exceeding max_pages raises ValueError."""
        mock_pdfinfo.return_value = {"Pages": 15}

        with pytest.raises(ValueError, match="exceeding the maximum"):
            pdf_to_images(b"fake-pdf-bytes", max_pages=10)

        # convert_from_bytes should not be called when page limit exceeded
        mock_convert.assert_not_called()

    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    @patch("app.services.pdf_processor.convert_from_bytes")
    def test_exact_page_limit_allowed(self, mock_convert, mock_pdfinfo):
        """PDF with exactly max_pages pages is allowed."""
        mock_pdfinfo.return_value = {"Pages": 10}
        pages = [_make_pil_image() for _ in range(10)]
        mock_convert.return_value = pages

        result = pdf_to_images(b"fake-pdf-bytes", max_pages=10)

        assert len(result) == 10

    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    @patch("app.services.pdf_processor.convert_from_bytes")
    def test_custom_dpi(self, mock_convert, mock_pdfinfo):
        """Custom DPI is passed to pdf2image."""
        mock_pdfinfo.return_value = {"Pages": 1}
        mock_convert.return_value = [_make_pil_image()]

        pdf_to_images(b"fake-pdf-bytes", dpi=150)

        mock_convert.assert_called_once_with(b"fake-pdf-bytes", dpi=150)

    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    @patch("app.services.pdf_processor.convert_from_bytes")
    def test_custom_max_pages(self, mock_convert, mock_pdfinfo):
        """Custom max_pages enforces the specified limit."""
        mock_pdfinfo.return_value = {"Pages": 6}

        with pytest.raises(ValueError):
            pdf_to_images(b"fake-pdf-bytes", max_pages=5)

    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    @patch("app.services.pdf_processor.convert_from_bytes")
    def test_rgb_to_bgr_conversion(self, mock_convert, mock_pdfinfo):
        """PIL RGB images are converted to BGR numpy arrays."""
        mock_pdfinfo.return_value = {"Pages": 1}
        # Create a known RGB image (red pixel)
        red_rgb = Image.fromarray(
            np.array([[[255, 0, 0]]], dtype=np.uint8), mode="RGB"
        )
        mock_convert.return_value = [red_rgb]

        result = pdf_to_images(b"fake-pdf-bytes")

        # In BGR, red should be [0, 0, 255]
        assert result[0][0, 0, 0] == 0    # B
        assert result[0][0, 0, 1] == 0    # G
        assert result[0][0, 0, 2] == 255  # R

    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    @patch("app.services.pdf_processor.convert_from_bytes")
    def test_pdfinfo_fallback(self, mock_convert, mock_pdfinfo):
        """Falls back to conversion-based counting when pdfinfo fails."""
        mock_pdfinfo.side_effect = Exception("poppler not found")
        pages = [_make_pil_image() for _ in range(3)]
        # First call is the fallback count (low DPI), second is actual conversion
        mock_convert.side_effect = [pages, pages]

        result = pdf_to_images(b"fake-pdf-bytes", max_pages=5)

        assert len(result) == 3


# ---------------------------------------------------------------------------
# Unit Tests: merge_ocr_results
# ---------------------------------------------------------------------------


class TestMergeOcrResults:
    """Unit tests for merge_ocr_results function."""

    def test_single_result_returned_as_is(self):
        """Single OCRResult is returned directly."""
        result = _make_ocr_result("hello world")
        merged = merge_ocr_results([result])

        assert merged is result

    def test_empty_list_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty list"):
            merge_ocr_results([])

    def test_merges_text_from_multiple_pages(self):
        """Text from multiple pages is concatenated."""
        r1 = _make_ocr_result("page one text")
        r2 = _make_ocr_result("page two text")

        merged = merge_ocr_results([r1, r2])

        assert "page one text" in merged.text
        assert "page two text" in merged.text

    def test_deduplicates_identical_text(self):
        """Identical text lines across pages are deduplicated."""
        r1 = _make_ocr_result("GOVERNMENT OF INDIA\nName: John")
        r2 = _make_ocr_result("GOVERNMENT OF INDIA\nAddress: Delhi")

        merged = merge_ocr_results([r1, r2])

        # "GOVERNMENT OF INDIA" should appear only once
        assert merged.text.count("GOVERNMENT OF INDIA") == 1
        assert "Name: John" in merged.text
        assert "Address: Delhi" in merged.text

    def test_confidence_aggregation(self):
        """Average confidence is computed across all pages."""
        r1 = _make_ocr_result(
            "text1",
            confidences=[0.8, 0.9],
        )
        r2 = _make_ocr_result(
            "text2",
            confidences=[0.7, 0.6],
        )

        merged = merge_ocr_results([r1, r2])

        expected_avg = (0.8 + 0.9 + 0.7 + 0.6) / 4
        assert abs(merged.avg_confidence - expected_avg) < 1e-9

    def test_bounding_boxes_concatenated(self):
        """Bounding boxes from all pages are concatenated."""
        box1 = [[0, 0], [50, 0], [50, 20], [0, 20]]
        box2 = [[10, 10], [60, 10], [60, 30], [10, 30]]
        box3 = [[20, 20], [70, 20], [70, 40], [20, 40]]

        r1 = _make_ocr_result("a", boxes=[box1], confidences=[0.9])
        r2 = _make_ocr_result("b", boxes=[box2, box3], confidences=[0.8, 0.7])

        merged = merge_ocr_results([r1, r2])

        assert len(merged.boxes) == 3
        assert merged.boxes[0] == box1
        assert merged.boxes[1] == box2
        assert merged.boxes[2] == box3

    def test_confidences_list_concatenated(self):
        """All individual confidence scores are preserved in the merged result."""
        r1 = _make_ocr_result("x", confidences=[0.95, 0.85])
        r2 = _make_ocr_result("y", confidences=[0.75])

        merged = merge_ocr_results([r1, r2])

        assert merged.confidences == [0.95, 0.85, 0.75]

    def test_empty_text_pages_handled(self):
        """Pages with empty text don't add blank lines."""
        r1 = _make_ocr_result("content here", confidences=[0.9])
        r2 = _make_ocr_result("", boxes=[], confidences=[])

        merged = merge_ocr_results([r1, r2])

        assert "content here" in merged.text
        assert merged.text.strip() == "content here"


# ---------------------------------------------------------------------------
# Property-Based Tests: PDF Page Limit Enforcement (Property 19)
# ---------------------------------------------------------------------------


class TestPdfPageLimitProperty:
    """Property 19: PDF page limit enforcement.

    *For any* PDF document with more than max_pages pages,
    the System SHALL raise ValueError (rejecting it).

    *For any* PDF document with pages <= max_pages,
    the System SHALL process it without error.

    **Validates: Requirements 10.2**
    """

    @given(
        page_count=st.integers(min_value=11, max_value=100),
        max_pages=st.just(10),
    )
    @settings(max_examples=100, deadline=500)
    @patch("app.services.pdf_processor.convert_from_bytes")
    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    def test_exceeding_page_limit_always_raises(
        self, mock_pdfinfo, mock_convert, page_count: int, max_pages: int
    ):
        """For ANY page_count > max_pages, pdf_to_images MUST raise ValueError."""
        mock_pdfinfo.return_value = {"Pages": page_count}

        with pytest.raises(ValueError):
            pdf_to_images(b"fake-pdf", max_pages=max_pages)

        # Conversion should never be attempted for over-limit PDFs
        mock_convert.assert_not_called()

    @given(
        page_count=st.integers(min_value=1, max_value=10),
        max_pages=st.just(10),
    )
    @settings(max_examples=100, deadline=500)
    @patch("app.services.pdf_processor.convert_from_bytes")
    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    def test_within_page_limit_never_raises(
        self, mock_pdfinfo, mock_convert, page_count: int, max_pages: int
    ):
        """For ANY page_count <= max_pages, pdf_to_images MUST NOT raise."""
        mock_pdfinfo.return_value = {"Pages": page_count}
        mock_convert.return_value = [_make_pil_image() for _ in range(page_count)]

        result = pdf_to_images(b"fake-pdf", max_pages=max_pages)

        assert len(result) == page_count

    @given(
        page_count=st.integers(min_value=1, max_value=200),
        max_pages=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=200, deadline=500)
    @patch("app.services.pdf_processor.convert_from_bytes")
    @patch("app.services.pdf_processor.pdfinfo_from_bytes")
    def test_page_limit_boundary_property(
        self, mock_pdfinfo, mock_convert, page_count: int, max_pages: int
    ):
        """For ANY page_count and max_pages:
        - If page_count > max_pages → ValueError raised
        - If page_count <= max_pages → no error, returns page_count images
        """
        mock_pdfinfo.return_value = {"Pages": page_count}
        mock_convert.return_value = [_make_pil_image() for _ in range(page_count)]

        if page_count > max_pages:
            with pytest.raises(ValueError):
                pdf_to_images(b"fake-pdf", max_pages=max_pages)
        else:
            result = pdf_to_images(b"fake-pdf", max_pages=max_pages)
            assert len(result) == page_count
