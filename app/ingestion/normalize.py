"""Normalization orchestrator: maps heterogeneous source files into canonical
``ProductRecord``, ``SupplierRecord``, ``TechnicalBulletin``, and raw text/OCR
pages ready for chunking.

Responsibilities (Section 4.2 – app.ingestion.normalize):
- Route files to the correct parser.
- Invoke OCR pipeline for ``needs_ocr`` pages.
- Apply text cleaning and injection detection.
- Emit cleaned, typed records.
- NO chunking, embedding, or vector-store access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.schemas import ProductRecord, SupplierRecord, TechnicalBulletin
from app.ingestion import cleaning, parsers
from app.ingestion.ocr.engine import OCRResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------


@dataclass
class NormalizedPage:
    """A cleaned, parsed page ready for the chunking stage."""

    page_number: int  # 1-indexed
    text: str  # Cleaned text (from direct extraction or OCR)
    source_file: str  # Originating filename
    source_type: str  # "extracted" | "ocr" | "ocr_vision"
    ocr_confidence: float | None  # None for non-OCR pages
    suspicious: bool = False


@dataclass
class NormalizedDocument:
    """All normalized outputs produced from a single source file."""

    source_file: str

    product_records: list[ProductRecord] = field(default_factory=list)
    supplier_records: list[SupplierRecord] = field(default_factory=list)
    bulletin_records: list[TechnicalBulletin] = field(default_factory=list)

    # Text sections (from Markdown/TXT)
    sections: list[parsers.Section] = field(default_factory=list)

    # PDF pages (from PDF/OCR pipeline)
    pages: list[NormalizedPage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalizer class
# ---------------------------------------------------------------------------


class DocumentNormalizer:
    """Orchestrates parsing, OCR, and cleaning for all supported file types.

    Args:
        ocr_engine: Optional ``TesseractEngine`` instance.  When *None*,
                    PDF pages flagged for OCR are skipped with a warning.
        vision_fallback: Optional ``VisionOCRFallback`` instance.
        ocr_dpi: Render resolution for PDF → image conversion (default 300).
        ocr_conf_threshold: Confidence below which vision fallback is triggered.
        ocr_min_text_chars: Char-count threshold for flagging pages as scanned.
    """

    def __init__(
        self,
        ocr_engine: Any | None = None,
        vision_fallback: Any | None = None,
        ocr_dpi: int = 300,
        ocr_conf_threshold: float = 0.60,
        ocr_min_text_chars: int = 50,
    ) -> None:
        self._ocr = ocr_engine
        self._vision = vision_fallback
        self._dpi = ocr_dpi
        self._conf_threshold = ocr_conf_threshold
        self._min_chars = ocr_min_text_chars

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def normalize(self, path: Path) -> NormalizedDocument:
        """Parse and clean a single source file.

        Args:
            path: Absolute or relative path to the source file.

        Returns:
            :class:`NormalizedDocument` populated with typed records and/or
            cleaned text pages.

        Raises:
            ValueError: For unsupported file extensions.
        """
        suffix = path.suffix.lower()

        if suffix == ".json":
            return self._from_parsed(parsers.parse_json(path))
        elif suffix == ".csv":
            return self._from_parsed(parsers.parse_csv(path))
        elif suffix in (".md", ".markdown"):
            return self._from_parsed(parsers.parse_markdown(path))
        elif suffix == ".txt":
            return self._from_parsed(parsers.parse_txt(path))
        elif suffix == ".pdf":
            return self._normalize_pdf(path)
        else:
            raise ValueError(
                f"Unsupported file extension '{suffix}' for file '{path.name}'. "
                "Supported: .json, .csv, .md, .markdown, .txt, .pdf"
            )

    # ------------------------------------------------------------------
    # Internal: structured / text document flow
    # ------------------------------------------------------------------

    def _from_parsed(self, parsed: parsers.ParsedDocument) -> NormalizedDocument:
        """Convert a ``ParsedDocument`` to a ``NormalizedDocument``.

        Applies text cleaning to sections. Product/supplier/bulletin records
        are stored as-is (they are already typed Pydantic objects).
        """
        doc = NormalizedDocument(source_file=parsed.source_file)
        doc.product_records = parsed.product_records
        doc.supplier_records = parsed.supplier_records

        # Clean bulletin content
        cleaned_bulletins: list[TechnicalBulletin] = []
        for b in parsed.bulletin_records:
            cleaned_content, suspicious = cleaning.clean(b.content)
            if suspicious:
                logger.warning(
                    "Prompt-injection detected in bulletin '%s'.", b.bulletin_id
                )
            cleaned_bulletins.append(
                TechnicalBulletin(
                    bulletin_id=b.bulletin_id,
                    title=b.title,
                    related_product_ids=b.related_product_ids,
                    content=cleaned_content,
                    source_document=b.source_document,
                )
            )
        doc.bulletin_records = cleaned_bulletins

        # Clean section bodies
        cleaned_sections: list[parsers.Section] = []
        for sec in parsed.sections:
            cleaned_body, _ = cleaning.clean(sec.body)
            cleaned_sections.append(
                parsers.Section(
                    heading=sec.heading,
                    body=cleaned_body,
                    level=sec.level,
                )
            )
        doc.sections = cleaned_sections
        return doc

    # ------------------------------------------------------------------
    # Internal: PDF flow with OCR
    # ------------------------------------------------------------------

    def _normalize_pdf(self, path: Path) -> NormalizedDocument:
        """Parse a PDF, routing scanned pages through OCR."""
        parsed = parsers.parse_pdf(path, ocr_min_text_chars=self._min_chars)
        doc = NormalizedDocument(source_file=parsed.source_file)

        for page_content in parsed.pages:
            if not page_content.needs_ocr:
                # Direct text extraction path
                cleaned_text, suspicious = cleaning.clean(page_content.text)
                doc.pages.append(
                    NormalizedPage(
                        page_number=page_content.page_number,
                        text=cleaned_text,
                        source_file=parsed.source_file,
                        source_type="extracted",
                        ocr_confidence=None,
                        suspicious=suspicious,
                    )
                )
            else:
                # OCR path
                normalized_page = self._ocr_page(path, page_content)
                doc.pages.append(normalized_page)

        return doc

    def _ocr_page(
        self, pdf_path: Path, page_content: parsers.PageContent
    ) -> NormalizedPage:
        """Render a PDF page to image, preprocess, and run OCR."""
        if self._ocr is None:
            logger.warning(
                "No OCR engine configured; skipping page %d of '%s'.",
                page_content.page_number,
                pdf_path.name,
            )
            return NormalizedPage(
                page_number=page_content.page_number,
                text="",
                source_file=pdf_path.name,
                source_type="ocr",
                ocr_confidence=None,
                suspicious=False,
            )

        # Render PDF page to PIL image
        image_array = self._render_pdf_page(pdf_path, page_content.page_number)

        # Preprocess
        from app.ingestion.ocr.preprocess import preprocess

        preprocessed = preprocess(image_array)

        # Run Tesseract
        ocr_result: OCRResult = self._ocr.run(preprocessed)

        # Vision fallback if confidence is below threshold
        source_type = "ocr"
        suspicious = False
        used_vision = False

        if ocr_result.mean_confidence < self._conf_threshold and self._vision is not None:
            vision_result, used_vision = self._vision.run(
                preprocessed,
                tesseract_fallback_text=ocr_result.text,
            )
            if used_vision:
                ocr_result = vision_result
                source_type = "ocr_vision"
            else:
                # Graceful degradation: suspicious flag set by vision module
                suspicious = True
        elif ocr_result.mean_confidence < self._conf_threshold:
            suspicious = True

        cleaned_text, injection_found = cleaning.clean(ocr_result.text)
        suspicious = suspicious or injection_found

        return NormalizedPage(
            page_number=page_content.page_number,
            text=cleaned_text,
            source_file=pdf_path.name,
            source_type=source_type,
            ocr_confidence=ocr_result.mean_confidence if source_type != "ocr_vision" else None,
            suspicious=suspicious,
        )

    def _render_pdf_page(self, pdf_path: Path, page_number: int) -> np.ndarray:
        """Render a single PDF page to a numpy uint8 BGR array at ``self._dpi``."""
        import fitz  # PyMuPDF

        pdf = fitz.open(str(pdf_path))
        page = pdf[page_number - 1]  # 0-indexed in PyMuPDF
        matrix = fitz.Matrix(self._dpi / 72, self._dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)  # type: ignore[attr-defined]
        pdf.close()

        import numpy as np
        from PIL import Image

        pil_img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return np.array(pil_img)
