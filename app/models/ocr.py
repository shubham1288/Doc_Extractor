"""OCR result data models."""

from dataclasses import dataclass


@dataclass
class OCRResult:
    """Result of OCR text extraction from an image.

    Attributes:
        text: Full extracted text concatenated from all detected regions.
        boxes: Bounding box coordinates for each detected text region.
            Each box is a list of 4 corner points [[x,y], [x,y], [x,y], [x,y]].
        confidences: Per-region confidence scores (0.0 to 1.0).
        avg_confidence: Mean confidence across all detected regions.
    """

    text: str
    boxes: list[list[list[int]]]
    confidences: list[float]
    avg_confidence: float
