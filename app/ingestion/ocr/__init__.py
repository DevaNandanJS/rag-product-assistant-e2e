"""OCR sub-package: engine protocol, Tesseract implementation,
image preprocessing, vision fallback, and disk caching.
"""

from app.ingestion.ocr.engine import OCREngine, OCRResult
from app.ingestion.ocr.tesseract import TesseractEngine

__all__ = ["OCREngine", "OCRResult", "TesseractEngine"]
