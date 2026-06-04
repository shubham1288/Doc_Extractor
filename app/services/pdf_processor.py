"""PDF processing utilities for multi-page document handling.

Provides PDF to image conversion, page limit enforcement, and
multi-page OCR result merging with deduplication and confidence aggregation.
"""

import numpy as np
from numpy.typing import NDArray
from pdf2image import convert_from_bytes
from pdf2image.pdf2image import pdfinfo_from_bytes

from app.models.ocr import OCRResult


def pdf_to_images(
    file_bytes: bytes, dpi: int = 300, max_pages: int = 10
) -> list[NDArray]:
    """Convert PDF pages to images using pdf2image.

    Args:
        file_bytes: Raw PDF file bytes.
        dpi: Resolution for rendering pages (default 300 DPI).
        max_pages: Maximum number of pages allowed. Raises ValueError
            if the PDF exceeds this limit.

    Returns:
        List of numpy arrays in BGR format (OpenCV compatible),
        one per page.

    Raises:
        ValueError: If the PDF has more pages than max_pages.
    """
    # Determine page count before full conversion.
    # Use pdfinfo_from_bytes if available, otherwise convert at low DPI to count.
    try:
        info = pdfinfo_from_bytes(file_bytes)
        page_count = info.get("Pages", 0)
    except Exception:
        # Fallback: attempt conversion to determine page count
        # Convert all pages at low DPI just to count
        all_pages = convert_from_bytes(file_bytes, dpi=72)
        page_count = len(all_pages)

    if page_count > max_pages:
        raise ValueError(
            f"PDF has {page_count} pages, exceeding the maximum of {max_pages} pages"
        )

    # Convert pages at the requested DPI
    pil_images = convert_from_bytes(file_bytes, dpi=dpi)

    # Convert PIL images to numpy arrays in BGR format for OpenCV
    images: list[NDArray] = []
    for pil_img in pil_images:
        # PIL images are RGB, convert to BGR for OpenCV compatibility
        rgb_array = np.array(pil_img, dtype=np.uint8)
        if rgb_array.ndim == 3 and rgb_array.shape[2] == 3:
            bgr_array = rgb_array[:, :, ::-1].copy()
        else:
            bgr_array = rgb_array
        images.append(bgr_array)

    return images


def merge_ocr_results(results: list[OCRResult]) -> OCRResult:
    """Merge OCR results from multiple pages.

    Combines text from multiple pages, deduplicates identical text lines,
    aggregates confidence scores, and concatenates bounding boxes.

    Args:
        results: List of OCRResult objects from individual pages.

    Returns:
        A single merged OCRResult with deduplicated text, concatenated
        bounding boxes (from unique lines), and averaged confidence.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("Cannot merge empty list of OCR results")

    if len(results) == 1:
        return results[0]

    # Split each result's text into lines for deduplication
    seen_lines: set[str] = set()
    unique_lines: list[str] = []
    all_boxes: list[list[list[int]]] = []
    all_confidences: list[float] = []

    for result in results:
        # Split text into lines and deduplicate
        lines = result.text.split("\n") if result.text else []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen_lines:
                seen_lines.add(stripped)
                unique_lines.append(stripped)

        # Concatenate bounding boxes from all pages
        all_boxes.extend(result.boxes)

        # Collect all confidence scores for aggregation
        all_confidences.extend(result.confidences)

    # Build merged text from unique lines
    merged_text = "\n".join(unique_lines)

    # Compute average confidence across all pages
    avg_confidence = (
        sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    )

    return OCRResult(
        text=merged_text,
        boxes=all_boxes,
        confidences=all_confidences,
        avg_confidence=avg_confidence,
    )
