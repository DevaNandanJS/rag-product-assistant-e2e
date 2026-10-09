"""Tesseract OCR engine implementation.

Wraps ``pytesseract.image_to_data`` and produces an :class:`OCRResult`
with per-word confidence scores and a mean confidence value.

Responsibilities (Section 4.2 – app.ingestion.ocr):
- Execute Tesseract on a preprocessed image.
- Parse confidence data and compute the mean.
- No image preprocessing, chunking, or vector-store logic.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from app.ingestion.ocr.engine import OCRResult

logger = logging.getLogger(__name__)

# Lazy import guard – pytesseract is optional at import time.
try:
    import pytesseract
    from pytesseract import Output

    _TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TESSERACT_AVAILABLE = False


try:
    import pandas  # noqa: F401

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


class TesseractEngine:
    """Tesseract-based OCR engine satisfying the :class:`OCREngine` protocol.

    Args:
        cmd_path: Optional explicit path to the ``tesseract`` binary.
                  When empty, auto-discovers standard Windows install
                  path or locates it via ``PATH``.
        lang: Tesseract language pack identifier (e.g. ``"eng"``).
        config: Additional Tesseract configuration string passed via ``--psm``
                or other flags (e.g. ``"--psm 6"``).
    """

    def __init__(
        self,
        cmd_path: str = "",
        lang: str = "eng",
        config: str = "--psm 6",
    ) -> None:
        if not _TESSERACT_AVAILABLE:
            raise ImportError(
                "pytesseract is not installed. "
                "Install it with: pip install pytesseract"
            )

        standard_win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        resolved = None

        if cmd_path and Path(cmd_path).exists():
            resolved = str(Path(cmd_path))
        elif standard_win_path.exists():
            resolved = str(standard_win_path)
        elif shutil.which("tesseract"):
            resolved = shutil.which("tesseract")

        if resolved:
            pytesseract.pytesseract.tesseract_cmd = resolved
            self._cmd_path = resolved
            logger.debug("Tesseract binary set to: %s", resolved)
        else:
            self._cmd_path = ""
            if cmd_path:
                logger.warning(
                    "Configured TESSERACT_CMD '%s' not found. "
                    "Falling back to PATH resolution.",
                    cmd_path,
                )

        self._lang = lang
        self._config = config

    @property
    def is_available(self) -> bool:
        """Return True if Tesseract binary was discovered and is executable."""
        if not self._cmd_path:
            return False
        return Path(self._cmd_path).exists() or bool(shutil.which(self._cmd_path))

    # ------------------------------------------------------------------
    # Public interface (satisfies OCREngine Protocol)
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray) -> OCRResult:
        """Run Tesseract on a preprocessed grayscale/binary image.

        Extracts word-level bounding boxes and confidence integers.
        Words with confidence == -1 are non-text layout items and
        are excluded from the mean.
        """
        output_type = Output.DATAFRAME if _PANDAS_AVAILABLE else Output.DICT
        try:
            raw_data = pytesseract.image_to_data(
                image,
                lang=self._lang,
                config=self._config,
                output_type=output_type,
            )
        except Exception as exc:
            logger.error("Tesseract execution failed: %s", exc)
            return OCRResult(text="", mean_confidence=0.0, source="tesseract")

        word_confs: list[float] = []

        if hasattr(raw_data, "iterrows"):
            word_rows = raw_data[(raw_data["conf"] >= 0) & (raw_data["text"].notna())]
            word_rows = word_rows[word_rows["text"].str.strip() != ""]
            for _, row in word_rows.iterrows():
                word_confs.append(float(row["conf"]) / 100.0)
            full_text = self._reconstruct_text_from_df(raw_data)
        elif isinstance(raw_data, dict):
            n_boxes = len(raw_data.get("text", []))
            for i in range(n_boxes):
                text = str(raw_data["text"][i]).strip()
                try:
                    conf = float(raw_data["conf"][i])
                except (ValueError, TypeError):
                    conf = -1.0
                if conf >= 0 and text:
                    word_confs.append(conf / 100.0)
            full_text = self._reconstruct_text_from_dict(raw_data)
        else:
            return OCRResult(text="", mean_confidence=0.0, source="tesseract")

        if not word_confs:
            return OCRResult(
                text="",
                mean_confidence=0.0,
                source="tesseract",
                word_confidences=[],
            )

        mean_conf = float(sum(word_confs) / len(word_confs))
        return OCRResult(
            text=full_text,
            mean_confidence=mean_conf,
            source="tesseract",
            word_confidences=word_confs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_text_from_df(df) -> str:  # type: ignore[no-untyped-def]
        """Reconstruct page text preserving line breaks from DataFrame TSV data."""
        lines: list[str] = []
        current_line_words: list[str] = []
        prev_line_num = None
        prev_block_num = None

        for _, row in df.iterrows():
            conf = row.get("conf", -1)
            text = str(row.get("text", "")).strip()
            block_num = row.get("block_num", 0)
            line_num = row.get("line_num", 0)

            if conf == -1 or not text:
                continue

            line_changed = (line_num != prev_line_num) or (block_num != prev_block_num)
            if line_changed and current_line_words:
                lines.append(" ".join(current_line_words))
                current_line_words = []

            current_line_words.append(text)
            prev_line_num = line_num
            prev_block_num = block_num

        if current_line_words:
            lines.append(" ".join(current_line_words))

        return "\n".join(lines)

    @staticmethod
    def _reconstruct_text_from_dict(data: dict) -> str:  # type: ignore[type-arg]
        """Reconstruct page text preserving line breaks from Output.DICT TSV data."""
        lines: list[str] = []
        current_line_words: list[str] = []
        prev_line_num = None
        prev_block_num = None
        n_boxes = len(data.get("text", []))

        for i in range(n_boxes):
            try:
                conf = float(data.get("conf", [])[i])
            except (ValueError, TypeError, IndexError):
                conf = -1.0
            text = str(data.get("text", [])[i]).strip()
            block_num = data.get("block_num", [0])[i] if i < len(data.get("block_num", [])) else 0
            line_num = data.get("line_num", [0])[i] if i < len(data.get("line_num", [])) else 0

            if conf == -1 or not text:
                continue

            line_changed = (line_num != prev_line_num) or (block_num != prev_block_num)
            if line_changed and current_line_words:
                lines.append(" ".join(current_line_words))
                current_line_words = []

            current_line_words.append(text)
            prev_line_num = line_num
            prev_block_num = block_num

        if current_line_words:
            lines.append(" ".join(current_line_words))

        return "\n".join(lines)
