"""Tests for the OCR engine singleton pattern and functionality."""

import threading
from unittest.mock import patch

import numpy as np
import pytest

from app.models.ocr import OCRResult
from app.ocr.engine import OCREngine, _LANG_MAP


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset OCREngine singleton before and after each test."""
    OCREngine.reset()
    yield
    OCREngine.reset()


@patch("app.ocr.engine._PADDLE_AVAILABLE", False)
class TestOCREngineSingleton:
    """Tests for thread-safe singleton behavior."""

    def test_singleton_returns_same_instance(self):
        """Creating two instances returns the same object."""
        engine1 = OCREngine()
        engine2 = OCREngine()
        assert engine1 is engine2

    def test_singleton_thread_safety(self):
        """Creating instances from multiple threads returns the same object."""
        instances: list[OCREngine] = []
        barrier = threading.Barrier(10)

        def create_instance():
            barrier.wait()
            instances.append(OCREngine())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads must have received the exact same instance.
        assert len(instances) == 10
        assert all(inst is instances[0] for inst in instances)

    def test_reset_allows_new_instance(self):
        """reset() allows creating a new instance."""
        engine1 = OCREngine()
        OCREngine.reset()
        engine2 = OCREngine()
        assert engine1 is not engine2


@patch("app.ocr.engine._PADDLE_AVAILABLE", False)
class TestOCREngineGracefulDegradation:
    """Tests for graceful degradation when PaddleOCR is not available."""

    def test_is_ready_returns_false_when_paddle_unavailable(self):
        """is_ready returns False when PaddleOCR is not installed."""
        engine = OCREngine()
        assert engine.is_ready is False

    def test_extract_text_returns_empty_ocr_result_when_not_ready(self):
        """extract_text returns an empty OCRResult when engine is not ready."""
        engine = OCREngine()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = engine.extract_text(image)

        assert isinstance(result, OCRResult)
        assert result.text == ""
        assert result.boxes == []
        assert result.confidences == []
        assert result.avg_confidence == 0.0

    def test_extract_text_with_languages_returns_empty_when_not_ready(self):
        """extract_text with explicit languages still degrades gracefully."""
        engine = OCREngine()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = engine.extract_text(image, languages=["hi", "en"])

        assert isinstance(result, OCRResult)
        assert result.text == ""
        assert result.avg_confidence == 0.0

    def test_extract_batch_returns_correct_number_of_results(self):
        """extract_batch returns one OCRResult per input image."""
        engine = OCREngine()
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
            np.zeros((50, 50, 3), dtype=np.uint8),
        ]

        results = engine.extract_batch(images)

        assert len(results) == 3
        assert all(isinstance(r, OCRResult) for r in results)
        assert all(r.text == "" for r in results)

    def test_extract_batch_empty_list_returns_empty(self):
        """extract_batch with empty list returns empty list."""
        engine = OCREngine()

        results = engine.extract_batch([])

        assert results == []


class TestOCRResultDataclass:
    """Tests for the OCRResult dataclass."""

    def test_ocr_result_creation(self):
        """OCRResult can be created with all fields."""
        result = OCRResult(
            text="Hello World",
            boxes=[[[0, 0], [100, 0], [100, 30], [0, 30]]],
            confidences=[0.95],
            avg_confidence=0.95,
        )

        assert result.text == "Hello World"
        assert len(result.boxes) == 1
        assert len(result.confidences) == 1
        assert result.avg_confidence == 0.95

    def test_ocr_result_empty(self):
        """OCRResult can represent empty extraction."""
        result = OCRResult(text="", boxes=[], confidences=[], avg_confidence=0.0)

        assert result.text == ""
        assert result.boxes == []
        assert result.confidences == []
        assert result.avg_confidence == 0.0

    def test_ocr_result_multiple_boxes(self):
        """OCRResult handles multiple detected text regions."""
        result = OCRResult(
            text="Line 1\nLine 2\nLine 3",
            boxes=[
                [[0, 0], [100, 0], [100, 30], [0, 30]],
                [[0, 40], [100, 40], [100, 70], [0, 70]],
                [[0, 80], [100, 80], [100, 110], [0, 110]],
            ],
            confidences=[0.9, 0.85, 0.92],
            avg_confidence=0.89,
        )

        assert len(result.boxes) == 3
        assert len(result.confidences) == 3
        assert result.avg_confidence == pytest.approx(0.89)


class TestOCREngineLanguageMapping:
    """Tests for multi-language support configuration."""

    def test_language_map_contains_english(self):
        """Language map includes English."""
        assert "en" in _LANG_MAP
        assert _LANG_MAP["en"] == "en"

    def test_language_map_contains_all_indian_languages(self):
        """Language map includes all 9 Indian languages."""
        expected_languages = {"hi", "mr", "gu", "bn", "ta", "te", "kn", "ml", "pa"}
        assert expected_languages.issubset(set(_LANG_MAP.keys()))

    def test_language_map_has_10_entries(self):
        """Language map has exactly 10 entries (English + 9 Indian)."""
        assert len(_LANG_MAP) == 10


class TestOCREngineParseResult:
    """Tests for the _parse_result static method."""

    def test_parse_result_none_returns_empty(self):
        """_parse_result handles None input."""
        result = OCREngine._parse_result(None)
        assert result.text == ""
        assert result.boxes == []
        assert result.confidences == []
        assert result.avg_confidence == 0.0

    def test_parse_result_empty_list_returns_empty(self):
        """_parse_result handles empty list."""
        result = OCREngine._parse_result([])
        assert result.text == ""

    def test_parse_result_empty_inner_list_returns_empty(self):
        """_parse_result handles [[]] (no detections on page)."""
        result = OCREngine._parse_result([[]])
        assert result.text == ""

    def test_parse_result_valid_output(self):
        """_parse_result correctly parses PaddleOCR output format."""
        raw = [[
            [[[10, 5], [100, 5], [100, 25], [10, 25]], ("Hello", 0.95)],
            [[[10, 35], [100, 35], [100, 55], [10, 55]], ("World", 0.88)],
        ]]

        result = OCREngine._parse_result(raw)

        assert result.text == "Hello\nWorld"
        assert len(result.boxes) == 2
        assert result.confidences == [0.95, 0.88]
        assert result.avg_confidence == pytest.approx((0.95 + 0.88) / 2)

    def test_parse_result_single_detection(self):
        """_parse_result works with a single detected text region."""
        raw = [[
            [[[0, 0], [50, 0], [50, 20], [0, 20]], ("Test", 0.99)],
        ]]

        result = OCREngine._parse_result(raw)

        assert result.text == "Test"
        assert len(result.boxes) == 1
        assert result.avg_confidence == 0.99
