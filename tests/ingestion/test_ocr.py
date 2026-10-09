"""Unit and mock tests for the OCR pipeline, preprocessing, caching, and normalizer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.ingestion.cleaning import clean
from app.ingestion.normalize import DocumentNormalizer
from app.ingestion.ocr.cache import VisionOCRCache
from app.ingestion.ocr.engine import OCRResult
from app.ingestion.ocr.preprocess import (
    apply_clahe,
    deskew,
    otsu_threshold,
    pil_to_bgr,
    preprocess,
    to_grayscale,
)
from app.ingestion.ocr.tesseract import TesseractEngine
from app.ingestion.ocr.vision import VisionOCRFallback

RAW_DATA_DIR = Path("data/raw")
TRANSCRIPTION_PROMPT = (
    "Transcribe all text in this image exactly as written. "
    "Keep table rows as separate lines with ' | ' between cells."
)


class TestImagePreprocessing:
    """Tests for image preprocessing pipeline in app.ingestion.ocr.preprocess."""

    def test_pil_to_bgr_and_to_grayscale(self) -> None:
        pil_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        bgr = pil_to_bgr(pil_img)
        assert bgr.shape == (64, 64, 3)

        gray = to_grayscale(bgr)
        assert gray.shape == (64, 64)
        assert len(gray.shape) == 2

    def test_otsu_threshold(self) -> None:
        gray = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape((64, 64))
        binary = otsu_threshold(gray)
        assert binary.shape == (64, 64)
        unique_vals = set(np.unique(binary))
        assert unique_vals.issubset({0, 255})

    def test_apply_clahe(self) -> None:
        gray = np.full((64, 64), 128, dtype=np.uint8)
        enhanced = apply_clahe(gray)
        assert enhanced.shape == (64, 64)
        assert enhanced.dtype == np.uint8

    def test_deskew_near_horizontal(self) -> None:
        binary = np.full((100, 100), 255, dtype=np.uint8)
        binary[40:60, 20:80] = 0
        deskewed = deskew(binary)
        assert deskewed.shape == (100, 100)

    def test_full_preprocess_pipeline(self) -> None:
        pil_img = Image.new("RGB", (80, 80), color=(240, 240, 240))
        result = preprocess(pil_img)
        assert result.shape == (80, 80)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})


class TestVisionOCRCache:
    """Tests for SHA-256 JSONL cache in app.ingestion.ocr.cache."""

    def test_cache_miss_and_put(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "test_cache.jsonl"
        cache = VisionOCRCache(cache_file)

        img_bytes = b"fake_jpeg_image_bytes"
        prompt = "Transcribe text"
        model = "gemini-3.1-flash-lite"

        assert cache.get(img_bytes, prompt, model) is None

        cache.put(img_bytes, prompt, model, "Extracted text from image")
        assert cache.get(img_bytes, prompt, model) == "Extracted text from image"

    def test_cache_persistence_across_instances(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "persistent_cache.jsonl"
        cache1 = VisionOCRCache(cache_file)
        img_bytes = b"persisted_bytes"
        prompt = "Transcribe"
        model = "test-model"

        cache1.put(img_bytes, prompt, model, "Persisted Transcription")

        cache2 = VisionOCRCache(cache_file)
        assert cache2.get(img_bytes, prompt, model) == "Persisted Transcription"


class TestTesseractEngineMock:
    """Tests for TesseractEngine using mock data without pandas requirement."""

    def test_mock_image_to_data_confidence_and_text(self) -> None:
        mock_data = {
            "block_num": [1, 1, 1, 1],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 2, 2],
            "word_num": [1, 2, 1, 0],
            "conf": [90, 80, 70, -1],
            "text": ["CartonPro", "1200", "Sealer", ""],
        }

        with patch("pytesseract.image_to_data", return_value=mock_data):
            engine = TesseractEngine()
            dummy_img = np.zeros((50, 50), dtype=np.uint8)
            result = engine.run(dummy_img)

            assert isinstance(result, OCRResult)
            assert result.source == "tesseract"
            assert result.text == "CartonPro 1200\nSealer"
            assert pytest.approx(result.mean_confidence, 0.01) == 0.80
            assert len(result.word_confidences) == 3

    def test_mock_empty_ocr_result(self) -> None:
        empty_data = {
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "word_num": [0],
            "conf": [-1],
            "text": [""],
        }

        with patch("pytesseract.image_to_data", return_value=empty_data):
            engine = TesseractEngine()
            dummy_img = np.zeros((50, 50), dtype=np.uint8)
            result = engine.run(dummy_img)

            assert result.text == ""
            assert result.mean_confidence == 0.0
            assert result.word_confidences == []


class TestVisionOCRFallback:
    """Tests for VisionOCRFallback."""

    def test_degrades_gracefully_when_no_api_key(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "vision_cache.jsonl"
        fallback = VisionOCRFallback(
            api_key="",
            base_url="https://dummy.example.com",
            model="gemini-3.1-flash-lite",
            cache_path=str(cache_file),
        )

        dummy_img = np.zeros((50, 50), dtype=np.uint8)
        result, used_vision = fallback.run(
            dummy_img,
            tesseract_fallback_text="Fallback Tesseract text",
        )

        assert used_vision is False
        assert result.text == "Fallback Tesseract text"
        assert result.mean_confidence == 0.0
        assert result.source == "tesseract"

    def test_returns_cached_transcription_when_available(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "vision_cache.jsonl"
        fallback = VisionOCRFallback(
            api_key="",
            base_url="https://dummy.example.com",
            model="gemini-3.1-flash-lite",
            cache_path=str(cache_file),
        )

        dummy_img = np.zeros((50, 50), dtype=np.uint8)
        img_bytes = fallback._array_to_jpeg_bytes(dummy_img)
        fallback._cache.put(
            img_bytes,
            TRANSCRIPTION_PROMPT,
            "gemini-3.1-flash-lite",
            "Cached Vision Text",
        )

        result, used_vision = fallback.run(dummy_img, tesseract_fallback_text="Fallback Text")
        assert used_vision is True
        assert result.text == "Cached Vision Text"
        assert result.source == "vision"
        assert result.mean_confidence == 1.0

    def test_calls_live_api_and_caches_result(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "vision_cache.jsonl"
        fallback = VisionOCRFallback(
            api_key="valid_dummy_key",
            base_url="https://dummy.example.com",
            model="gemini-3.1-flash-lite",
            cache_path=str(cache_file),
        )

        dummy_img = np.zeros((50, 50), dtype=np.uint8)
        with patch.object(fallback, "_call_vision_api", return_value="Live Vision Transcription"):
            result, used_vision = fallback.run(dummy_img, tesseract_fallback_text="Fallback")
            assert used_vision is True
            assert result.text == "Live Vision Transcription"
            assert result.source == "vision"

            img_bytes = fallback._array_to_jpeg_bytes(dummy_img)
            cached_val = fallback._cache.get(
                img_bytes,
                TRANSCRIPTION_PROMPT,
                "gemini-3.1-flash-lite",
            )
            assert cached_val == "Live Vision Transcription"


class TestDocumentNormalizerPipeline:
    """End-to-end tests for DocumentNormalizer orchestrating parsers, OCR, and cleaning."""

    def test_normalizer_routes_catalog_json(self) -> None:
        catalog_path = RAW_DATA_DIR / "catalog.json"
        if not catalog_path.exists():
            pytest.skip(f"{catalog_path} not found")

        normalizer = DocumentNormalizer()
        doc = normalizer.normalize(catalog_path)
        assert len(doc.product_records) > 0
        assert doc.source_file == "catalog.json"

    def test_normalizer_cleans_and_flags_injected_bulletin(self) -> None:
        injected_path = RAW_DATA_DIR / "bulletins" / "SUPPLIER-ADV-INJECTED.md"
        if not injected_path.exists():
            pytest.skip(f"{injected_path} not found")

        normalizer = DocumentNormalizer()
        doc = normalizer.normalize(injected_path)
        assert len(doc.sections) > 0
        full_text = " ".join(s.body for s in doc.sections)
        _, suspicious = clean(full_text)
        assert suspicious is True

    def test_normalizer_native_pdf_no_ocr(self) -> None:
        pdf_path = RAW_DATA_DIR / "PKG-120_datasheet.pdf"
        if not pdf_path.exists():
            pytest.skip(f"{pdf_path} not found")

        normalizer = DocumentNormalizer()
        doc = normalizer.normalize(pdf_path)
        assert len(doc.pages) > 0
        for page in doc.pages:
            assert page.source_type == "extracted"
            assert page.ocr_confidence is None
            assert len(page.text) > 0

    def test_normalizer_scanned_pdf_with_mock_ocr(self) -> None:
        pdf_path = RAW_DATA_DIR / "WHS-1800_scanned.pdf"
        if not pdf_path.exists():
            pytest.skip(f"{pdf_path} not found")

        mock_ocr = MagicMock()
        mock_ocr.run.return_value = OCRResult(
            text="Mock scanned text for WHS-1800 pallet rack",
            mean_confidence=0.88,
            source="tesseract",
        )

        normalizer = DocumentNormalizer(ocr_engine=mock_ocr)
        doc = normalizer.normalize(pdf_path)
        assert len(doc.pages) > 0
        for page in doc.pages:
            assert page.source_type == "ocr"
            assert page.ocr_confidence == 0.88
            assert "WHS-1800" in page.text


@pytest.mark.needs_tesseract
class TestTesseractEngineReal:
    """Live Tesseract execution tests (skipped if tesseract not installed)."""

    def test_real_tesseract_on_spec_plate(self) -> None:
        plate_path = RAW_DATA_DIR / "REF-320_spec_plate.png"
        if not plate_path.exists():
            pytest.skip(f"{plate_path} not found")

        engine = TesseractEngine()
        if not engine.is_available:
            pytest.skip("Tesseract OCR binary not found on this system")

        img = Image.open(plate_path)
        preprocessed = preprocess(img)
        result = engine.run(preprocessed)

        assert len(result.text.strip()) > 0
        assert result.mean_confidence > 0.0
