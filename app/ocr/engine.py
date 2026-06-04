"""Thread-safe singleton OCR engine wrapping PaddleOCR.

This module provides a single shared OCR engine instance for the application.
PaddleOCR models are loaded once and reused across all requests. The engine
gracefully degrades if PaddleOCR is not installed — ``is_ready`` will return
False and extraction methods will return empty results.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from app.core.config import get_settings
from app.models.ocr import OCRResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Attempt to import PaddleOCR; allow graceful degradation.
try:
    from paddleocr import PaddleOCR  # type: ignore[import-untyped]

    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Language code mapping: config codes → PaddleOCR language identifiers.
_LANG_MAP: dict[str, str] = {
    "en": "en",
    "hi": "hi",
    "mr": "mr",
    "gu": "gu",
    "bn": "bn",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
    "pa": "pa",
}


class OCREngine:
    """Singleton PaddleOCR wrapper with thread safety.

    Only one instance is ever created. All calls to ``OCREngine()`` return the
    same object. The underlying PaddleOCR model is loaded on first
    instantiation.
    """

    _instance: OCREngine | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> OCREngine:
        """Thread-safe singleton instantiation using double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                # Double-check after acquiring the lock.
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the engine (runs only once due to _initialized guard)."""
        if self._initialized:
            return
        self._initialized = True
        self._model: object | None = None
        self._inference_lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        """Load PaddleOCR models (CPU-optimized, MKL-DNN enabled).

        If PaddleOCR is not available, the engine stays in a degraded state
        where ``is_ready`` returns False.
        """
        if not _PADDLE_AVAILABLE:
            logger.warning(
                "PaddleOCR is not installed. OCR engine running in degraded mode."
            )
            return

        settings = get_settings()
        try:
            self._model = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=settings.ocr_use_gpu,
                enable_mkldnn=True,
                cpu_threads=4,
                show_log=False,
            )
            logger.info("PaddleOCR engine initialized successfully (CPU mode, MKL-DNN enabled).")
        except Exception:
            logger.exception("Failed to initialize PaddleOCR engine.")
            self._model = None

    @property
    def is_ready(self) -> bool:
        """Check if the OCR model is loaded and ready for inference."""
        return self._model is not None

    def extract_text(
        self,
        image: NDArray[np.uint8],
        languages: list[str] | None = None,
    ) -> OCRResult:
        """Extract text from a preprocessed image.

        Args:
            image: Preprocessed image as a numpy uint8 array.
            languages: Optional list of language codes to use. Falls back to
                settings.ocr_languages if not provided.

        Returns:
            OCRResult containing extracted text, bounding boxes, confidence
            scores, and average confidence.
        """
        if not self.is_ready:
            return OCRResult(text="", boxes=[], confidences=[], avg_confidence=0.0)

        # Resolve language configuration (PaddleOCR uses a single lang per call;
        # we use the first requested language or default to English).
        if languages:
            lang = _LANG_MAP.get(languages[0], "en")
        else:
            settings = get_settings()
            lang = _LANG_MAP.get(settings.ocr_languages[0], "en") if settings.ocr_languages else "en"

        # Run OCR with thread safety.
        with self._inference_lock:
            try:
                result = self._model.ocr(image, cls=True)  # type: ignore[union-attr]
            except Exception:
                logger.exception("OCR inference failed.")
                return OCRResult(text="", boxes=[], confidences=[], avg_confidence=0.0)

        return self._parse_result(result)

    def extract_batch(self, images: list[NDArray[np.uint8]]) -> list[OCRResult]:
        """Process multiple images sequentially.

        Args:
            images: List of preprocessed images.

        Returns:
            List of OCRResult objects, one per input image.
        """
        return [self.extract_text(img) for img in images]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_result(raw_result: list | None) -> OCRResult:
        """Parse PaddleOCR output into an OCRResult dataclass.

        PaddleOCR returns a nested list structure:
            [ [ [box, (text, confidence)], ... ] ]
        """
        if not raw_result or not raw_result[0]:
            return OCRResult(text="", boxes=[], confidences=[], avg_confidence=0.0)

        texts: list[str] = []
        boxes: list[list[list[int]]] = []
        confidences: list[float] = []

        for line in raw_result[0]:
            box_coords, (text, confidence) = line
            texts.append(text)
            # Convert box coordinates to int lists.
            boxes.append([[int(pt[0]), int(pt[1])] for pt in box_coords])
            confidences.append(float(confidence))

        full_text = "\n".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            boxes=boxes,
            confidences=confidences,
            avg_confidence=avg_confidence,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset singleton state, allowing a new instance to be created.

        Intended for use in tests only.
        """
        with cls._lock:
            cls._instance = None

    # Keep backward-compatible alias.
    _reset_singleton = reset
