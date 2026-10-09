"""Vision LLM fallback OCR handler.

Triggered when Tesseract mean confidence falls below the configured threshold
(``Settings.OCR_CONF_THRESHOLD``, default 0.60). Calls an OpenAI-compatible
vision API with a fixed transcription prompt, caches results by SHA-256, and
degrades gracefully (returns the Tesseract text with ``suspicious=True``) when
no API key is configured.

Responsibilities (Section 4.2 – app.ingestion.ocr):
- Encode image, call vision API, cache result.
- No chunking, vector-store operations, or prompt engineering for retrieval.
"""

from __future__ import annotations

import base64
import io
import logging

import numpy as np
from PIL import Image

from app.ingestion.ocr.cache import VisionOCRCache
from app.ingestion.ocr.engine import OCRResult

logger = logging.getLogger(__name__)

_TRANSCRIPTION_PROMPT = (
    "Transcribe all text in this image exactly as written. "
    "Keep table rows as separate lines with ' | ' between cells."
)


class VisionOCRFallback:
    """Vision-model OCR fallback for low-confidence Tesseract results.

    Args:
        api_key: API key for the vision provider. Empty string disables live calls.
        base_url: OpenAI-compatible endpoint base URL.
        model: Vision-capable model identifier (e.g. ``"gemini-3.1-flash-lite"``).
        cache_path: Path for the JSONL cache file.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        cache_path: str,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._cache = VisionOCRCache(cache_path)
        self._available = bool(api_key and api_key not in {"", "your_key_here"})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        image: np.ndarray,
        tesseract_fallback_text: str = "",
    ) -> tuple[OCRResult, bool]:
        """Attempt vision OCR for the given preprocessed image.

        Returns a tuple of ``(OCRResult, used_vision)`` where ``used_vision``
        is *True* only when a live API call or cached vision result was used.

        When the vision API is unavailable (no key), returns the
        Tesseract text with ``suspicious=True`` as a graceful degradation.
        """
        image_bytes = self._array_to_jpeg_bytes(image)

        # 1. Check cache first (works even without an API key)
        cached = self._cache.get(image_bytes, _TRANSCRIPTION_PROMPT, self._model)
        if cached is not None:
            logger.debug("Vision OCR cache hit for image hash.")
            return (
                OCRResult(text=cached, mean_confidence=1.0, source="vision"),
                True,
            )

        # 2. If no key configured, degrade gracefully
        if not self._available:
            logger.warning(
                "Vision OCR fallback requested but no API key configured. "
                "Retaining Tesseract text and marking chunk as suspicious."
            )
            return (
                OCRResult(
                    text=tesseract_fallback_text,
                    mean_confidence=0.0,
                    source="tesseract",
                ),
                False,
            )

        # 3. Live API call
        transcription = self._call_vision_api(image_bytes)
        self._cache.put(image_bytes, _TRANSCRIPTION_PROMPT, self._model, transcription)
        return (
            OCRResult(text=transcription, mean_confidence=1.0, source="vision"),
            True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _array_to_jpeg_bytes(self, image: np.ndarray) -> bytes:
        """Convert a numpy image array to JPEG bytes for API transmission."""
        pil_img = Image.fromarray(image)
        # Convert binary/grayscale to RGB for JPEG compatibility
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _call_vision_api(self, image_bytes: bytes) -> str:
        """Make a synchronous call to the vision API and return transcribed text.

        Uses ``openai`` SDK pointing at the configured base URL.
        """
        try:
            from openai import OpenAI  # local import to avoid hard dep if unused
        except ImportError:
            logger.error("openai package not installed; vision fallback unavailable.")
            return ""

        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {"type": "text", "text": _TRANSCRIPTION_PROMPT},
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            return text.strip()
        except Exception as exc:
            logger.error("Vision OCR API call failed: %s", exc)
            return ""
