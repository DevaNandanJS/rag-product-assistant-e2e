"""Document parsers for all supported source formats.

Parsers are pure extraction functions: they read a file and return either
canonical domain records (``ProductRecord``, ``SupplierRecord``,
``TechnicalBulletin``) or a list of raw text sections ready for chunking.

Responsibilities (Section 4.2 – app.ingestion.parsers):
- JSON / CSV  → ProductRecord | SupplierRecord list
- Markdown / TXT → list of (heading, body_text) tuples
- PDF (native text) → list of (page_number, text) tuples
- PDF (scanned)   → flag pages for OCR via ``parse_pdf`` return signal
- No chunking, embedding, or vector-store calls.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.schemas import ProductRecord, SpecValue, SupplierRecord, TechnicalBulletin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supporting data structures
# ---------------------------------------------------------------------------


@dataclass
class PageContent:
    """Represents one page's extracted content from a PDF or image document."""

    page_number: int  # 1-indexed
    text: str  # Direct text extraction (empty string if none)
    needs_ocr: bool  # True when page should be routed through OCR
    image_ratio: float = 0.0  # Fraction of page area covered by embedded images


@dataclass
class Section:
    """A logical text section parsed from Markdown or TXT files."""

    heading: str  # Heading text; empty string for the document preamble
    body: str  # Section body content (without heading line)
    level: int = 1  # Heading level (1–6); 0 for preamble / flat text


@dataclass
class ParsedDocument:
    """Container returned by all parsers, unifying output shapes."""

    source_file: str  # Basename of the originating file

    # Populated by JSON/CSV parsers
    product_records: list[ProductRecord] = field(default_factory=list)
    supplier_records: list[SupplierRecord] = field(default_factory=list)
    bulletin_records: list[TechnicalBulletin] = field(default_factory=list)

    # Populated by Markdown/TXT parsers
    sections: list[Section] = field(default_factory=list)

    # Populated by PDF parser
    pages: list[PageContent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OCR_MIN_TEXT_CHARS_DEFAULT = 50
_OCR_IMAGE_AREA_RATIO_DEFAULT = 0.25  # 25 % image coverage triggers OCR

# ---------------------------------------------------------------------------
# JSON / CSV parsers
# ---------------------------------------------------------------------------


def parse_json(path: Path) -> ParsedDocument:
    """Parse a catalog JSON file into ``ProductRecord`` or ``SupplierRecord`` objects.

    The function inspects the top-level keys of each record to determine whether
    it represents a product (has ``product_id``) or a supplier (has ``supplier_id``
    but not ``product_id``).

    Args:
        path: Path to the ``.json`` file.

    Returns:
        :class:`ParsedDocument` with ``product_records`` and/or
        ``supplier_records`` populated.

    Raises:
        ValueError: When the file is not valid JSON or has an unrecognised schema.
    """
    doc = ParsedDocument(source_file=path.name)

    raw: Any
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc

    if isinstance(raw, dict):
        raw = [raw]  # single-object files treated as a list of one

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array or object, got {type(raw)} in '{path}'")

    for item in raw:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict item in '%s': %r", path.name, item)
            continue

        if "product_id" in item:
            doc.product_records.append(_dict_to_product_record(item, path.name))
        elif "supplier_id" in item:
            doc.supplier_records.append(_dict_to_supplier_record(item, path.name))
        else:
            logger.warning(
                "Skipping unrecognised JSON record in '%s' (no product_id or supplier_id).",
                path.name,
            )

    return doc


def parse_csv(path: Path) -> ParsedDocument:
    """Parse a CSV file into ``ProductRecord`` or ``SupplierRecord`` objects.

    The first row must be a header row. If a ``product_id`` column is present,
    records are mapped to ``ProductRecord``; if ``supplier_id`` is present
    (without ``product_id``), records are mapped to ``SupplierRecord``.

    Args:
        path: Path to the ``.csv`` file.

    Returns:
        :class:`ParsedDocument` with populated records.
    """
    doc = ParsedDocument(source_file=path.name)

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        logger.warning("CSV file '%s' is empty or has no data rows.", path.name)
        return doc

    for row in rows:
        if "product_id" in row:
            try:
                doc.product_records.append(_csv_row_to_product_record(row, path.name))
            except Exception as exc:
                logger.warning("Skipping invalid CSV row in '%s': %s", path.name, exc)
        elif "supplier_id" in row:
            try:
                doc.supplier_records.append(_csv_row_to_supplier_record(row, path.name))
            except Exception as exc:
                logger.warning("Skipping invalid CSV row in '%s': %s", path.name, exc)

    return doc


# ---------------------------------------------------------------------------
# Markdown / TXT parsers
# ---------------------------------------------------------------------------


def parse_markdown(path: Path) -> ParsedDocument:
    """Parse a Markdown file into a list of heading + body :class:`Section` objects.

    Sections are split on ATX headings (``# Heading``). The text before the
    first heading is captured as a *preamble* section with level 0. Empty
    body sections are retained so that heading structure is preserved.

    For bulletins, the caller may inspect ``Section.heading`` to detect the
    bulletin ID from the ``# Technical Advisory XXX`` title line.

    Args:
        path: Path to the ``.md`` file.

    Returns:
        :class:`ParsedDocument` with ``sections`` populated.
    """
    doc = ParsedDocument(source_file=path.name)

    text = path.read_text(encoding="utf-8")

    # Detect bulletin metadata from YAML-like header lines
    bulletin = _try_parse_bulletin_markdown(text, path.name)
    if bulletin is not None:
        doc.bulletin_records.append(bulletin)
        return doc

    # Generic markdown → sections
    doc.sections = _split_markdown_sections(text)
    return doc


def parse_txt(path: Path) -> ParsedDocument:
    """Parse a plain-text file into logical sections separated by blank lines.

    Unlike Markdown, TXT files have no heading syntax. The entire file is
    treated as a single section with an empty heading. Paragraphs separated
    by two or more blank lines are captured as distinct sections.

    Args:
        path: Path to the ``.txt`` file.

    Returns:
        :class:`ParsedDocument` with ``sections`` populated.
    """
    doc = ParsedDocument(source_file=path.name)
    text = path.read_text(encoding="utf-8")

    # Split on double blank lines to get paragraphs
    paragraphs = re.split(r"\n{2,}", text.strip())
    for para in paragraphs:
        stripped = para.strip()
        if stripped:
            doc.sections.append(Section(heading="", body=stripped, level=0))

    return doc


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------


def parse_pdf(
    path: Path,
    ocr_min_text_chars: int = _OCR_MIN_TEXT_CHARS_DEFAULT,
    ocr_image_area_ratio: float = _OCR_IMAGE_AREA_RATIO_DEFAULT,
) -> ParsedDocument:
    """Extract text from a PDF file, flagging scanned pages for OCR.

    Decision logic (hybrid OCR heuristic from Section 3 / Study Notes):
    1. If ``page.get_text()`` returns fewer than ``ocr_min_text_chars``
       characters → ``needs_ocr = True``.
    2. If embedded images cover ≥ ``ocr_image_area_ratio`` of the page area
       (and extracted text is sparse) → ``needs_ocr = True``.

    The caller (``app.ingestion.normalize``) routes ``needs_ocr=True`` pages
    through the OCR pipeline.

    Args:
        path: Path to the PDF file.
        ocr_min_text_chars: Character count threshold below which OCR is triggered.
        ocr_image_area_ratio: Fractional page-area threshold for image coverage.

    Returns:
        :class:`ParsedDocument` with ``pages`` populated, each carrying
        ``needs_ocr`` flags and raw extracted text.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF parsing. "
            "Install it with: pip install pymupdf"
        ) from exc

    doc = ParsedDocument(source_file=path.name)

    try:
        pdf = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{path}': {exc}") from exc

    for page_idx in range(len(pdf)):
        page = pdf[page_idx]
        page_number = page_idx + 1  # 1-indexed

        # --- Text extraction ---
        text: str = page.get_text()  # type: ignore[attr-defined]
        text_chars = len(text.strip())

        # --- Image area ratio ---
        page_area = page.rect.width * page.rect.height
        image_area = 0.0
        if page_area > 0:
            for img_info in page.get_images(full=True):  # type: ignore[attr-defined]
                # img_info: (xref, smask, w, h, bpc, cs, alt_cs, name, filter, ...)
                # Use bounding box of image instance on the page
                for img_rect in page.get_image_rects(img_info[0]):  # type: ignore[attr-defined]
                    image_area += abs(img_rect.width * img_rect.height)
            image_ratio = min(image_area / page_area, 1.0)
        else:
            image_ratio = 0.0

        # --- Decision ---
        needs_ocr = text_chars < ocr_min_text_chars or (
            image_ratio >= ocr_image_area_ratio and text_chars < ocr_min_text_chars * 2
        )

        doc.pages.append(
            PageContent(
                page_number=page_number,
                text=text,
                needs_ocr=needs_ocr,
                image_ratio=image_ratio,
            )
        )
        logger.debug(
            "PDF '%s' page %d: chars=%d, img_ratio=%.2f, needs_ocr=%s",
            path.name,
            page_number,
            text_chars,
            image_ratio,
            needs_ocr,
        )

    pdf.close()
    return doc


# ---------------------------------------------------------------------------
# Internal conversion helpers
# ---------------------------------------------------------------------------


def _dict_to_product_record(item: dict[str, Any], source: str) -> ProductRecord:
    """Convert a raw dict (from JSON) to a ``ProductRecord``."""
    specs: dict[str, SpecValue] = {}
    raw_specs = item.get("specs", {})
    if isinstance(raw_specs, dict):
        for key, val in raw_specs.items():
            if isinstance(val, dict):
                specs[key] = SpecValue(
                    value=val.get("value"),
                    unit=val.get("unit"),
                    raw=val.get("raw"),
                )
            else:
                specs[key] = SpecValue(value=val)

    return ProductRecord(
        product_id=str(item["product_id"]),
        product_name=str(item.get("product_name", "")),
        category=str(item.get("category", "")),
        supplier_id=item.get("supplier_id"),
        supplier_name=item.get("supplier_name"),
        country=item.get("country", "IN"),
        description=item.get("description"),
        specs=specs,
        source_document=item.get("source_document", source),
    )


def _dict_to_supplier_record(item: dict[str, Any], source: str) -> SupplierRecord:
    """Convert a raw dict (from JSON) to a ``SupplierRecord``."""
    return SupplierRecord(
        supplier_id=str(item["supplier_id"]),
        supplier_name=str(item.get("supplier_name", "")),
        headquarters=str(item.get("headquarters", "")),
        country=str(item.get("country", "IN")),
        categories=list(item.get("categories", [])),
        regions_served=list(item.get("regions_served", [])),
        lead_time_notes=item.get("lead_time_notes"),
        source_document=item.get("source_document", source),
    )


def _csv_row_to_product_record(row: dict[str, str], source: str) -> ProductRecord:
    """Convert a CSV row dict to a ``ProductRecord`` with basic spec mapping."""
    return ProductRecord(
        product_id=row["product_id"].strip(),
        product_name=row.get("product_name", "").strip(),
        category=row.get("category", "").strip(),
        supplier_id=row.get("supplier_id") or None,
        supplier_name=row.get("supplier_name") or None,
        country=row.get("country", "IN").strip() or "IN",
        description=row.get("description") or None,
        specs={},  # CSV rows don't carry nested spec dicts
        source_document=row.get("source_document", source),
    )


def _csv_row_to_supplier_record(row: dict[str, str], source: str) -> SupplierRecord:
    """Convert a CSV row dict to a ``SupplierRecord``."""
    categories_raw = row.get("categories", "")
    categories = [c.strip() for c in categories_raw.split(",") if c.strip()]

    regions_raw = row.get("regions_served", "")
    regions = [r.strip() for r in regions_raw.split(",") if r.strip()]

    return SupplierRecord(
        supplier_id=row["supplier_id"].strip(),
        supplier_name=row.get("supplier_name", "").strip(),
        headquarters=row.get("headquarters", "").strip(),
        country=row.get("country", "IN").strip() or "IN",
        categories=categories,
        regions_served=regions,
        lead_time_notes=row.get("lead_time_notes") or None,
        source_document=row.get("source_document", source),
    )


def _try_parse_bulletin_markdown(text: str, source: str) -> TechnicalBulletin | None:
    """Attempt to parse a Markdown file as a ``TechnicalBulletin``.

    Heuristic: the first ``# `` heading must contain a bulletin ID pattern
    (e.g. ``PR-TECH-07``, ``WHS-SAFE-04``). Returns None for non-bulletin files.
    """
    _BULLETIN_HEADING_RE = re.compile(
        r"^#\s+(?:Technical Advisory|Bulletin|Advisory)\s+([\w-]+)", re.MULTILINE | re.IGNORECASE
    )
    match = _BULLETIN_HEADING_RE.search(text)
    if not match:
        return None

    bulletin_id = match.group(1).strip()
    title = match.group(0).lstrip("# ").strip()

    # Extract related product IDs from "Related Products:" line
    related_re = re.compile(r"\*\*Related Products:\*\*\s*(.+)", re.IGNORECASE)
    related_match = related_re.search(text)
    related_ids: list[str] = []
    if related_match:
        for part in related_match.group(1).split(","):
            pid = part.strip()
            if pid:
                related_ids.append(pid)

    return TechnicalBulletin(
        bulletin_id=bulletin_id,
        title=title,
        related_product_ids=related_ids,
        content=text.strip(),
        source_document=source,
    )


def _split_markdown_sections(text: str) -> list[Section]:
    """Split markdown text into :class:`Section` objects on ATX headings."""
    heading_re = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

    sections: list[Section] = []
    matches = list(heading_re.finditer(text))

    if not matches:
        # No headings: treat entire text as one section
        return [Section(heading="", body=text.strip(), level=0)]

    # Capture preamble (text before first heading)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(heading="", body=preamble, level=0))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append(Section(heading=heading, body=body, level=level))

    return sections
