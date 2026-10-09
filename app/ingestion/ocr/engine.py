"""OCREngine protocol – defines the interface that all OCR implementations must satisfy.

Responsibilities (bounded by Section 4.2):
- Define the OCRResult dataclass and OCREngine Protocol.
- No chunking, vector-store, or prompt-engineering logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class OCRResult:
    """Structured result returned by any OCR engine implementation."""

    text: str
    """Concatenated page/block text extracted by the engine."""

    mean_confidence: float
    """Mean word-level confidence score in [0.0, 1.0].
    Words with raw confidence < 0 (layout items) are excluded from this average.
    """

    source: str = "tesseract"
    """Identifier of the engine that produced this result ('tesseract' or 'vision')."""

    word_confidences: list[float] = field(default_factory=list)
    """Per-word confidence values (scaled 0-1). Empty for vision-model results."""


@runtime_checkable
class OCREngine(Protocol):
    """Protocol that every OCR backend must satisfy."""

    def run(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a preprocessed single-channel or BGR image.

        Args:
            image: NumPy array with dtype uint8. Preprocessing (grayscale,
                   deskew, thresholding) is expected to have been applied
                   **before** this call by ``app.ingestion.ocr.preprocess``.

        Returns:
            An :class:`OCRResult` carrying extracted text and confidence.
        """
        ...
