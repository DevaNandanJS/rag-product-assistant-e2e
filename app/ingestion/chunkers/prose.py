"""Recursive semantic prose chunker.
Splits technical bulletins, markdown sections, and OCR pages by headings,
blank lines, and sentences, preserving table rows and prepending context headers.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

from app.core.schemas import Chunk
from app.ingestion.chunkers.tokens import TokenCounter, WhitespaceTokenCounter


def _build_context_header(
    product_name: str | None,
    category: str | None,
    section: str | None,
) -> str:
    """Build a context header like: '{product_name} | {category} | {section}'."""
    parts: list[str] = []
    if product_name:
        parts.append(product_name.strip())
    if category:
        parts.append(category.strip())
    if section:
        parts.append(section.strip())
    return " | ".join(parts)


def _split_into_blocks(text: str) -> list[str]:
    """Split text into semantic blocks while keeping markdown tables intact."""
    lines = text.splitlines()
    blocks: list[str] = []
    current_table: list[str] = []
    current_para: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Markdown table row check
        if "|" in stripped and (stripped.startswith("|") or stripped.endswith("|") or " | " in stripped):
            if current_para:
                blocks.append("\n".join(current_para))
                current_para = []
            current_table.append(line)
        else:
            if current_table:
                blocks.append("\n".join(current_table))
                current_table = []
            if not stripped:
                if current_para:
                    blocks.append("\n".join(current_para))
                    current_para = []
            else:
                current_para.append(line)

    if current_table:
        blocks.append("\n".join(current_table))
    if current_para:
        blocks.append("\n".join(current_para))

    return [b.strip() for b in blocks if b.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences."""
    # Split on period/question/exclamation followed by space or newline
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in sentences if s.strip()]


def chunk_prose(
    text: str,
    document: str,
    page: int | None = None,
    section: str | None = None,
    product_id: str | None = None,
    product_name: str | None = None,
    category: str | None = None,
    supplier_id: str | None = None,
    supplier_name: str | None = None,
    country: str | None = None,
    source_type: Literal["structured", "extracted", "ocr", "ocr_vision"] = "extracted",
    ocr_confidence: float | None = None,
    suspicious: bool = False,
    counter: TokenCounter | None = None,
    target_tokens: int = 250,
    max_tokens: int = 480,
) -> list[Chunk]:
    """Recursively split prose text into token-bounded chunks with context headers.

    Args:
        text: Input prose text.
        document: Originating filename.
        page: Optional 1-indexed page number.
        section: Optional section heading.
        product_id: Associated product identifier.
        product_name: Associated product name.
        category: Associated category name.
        supplier_id: Associated supplier identifier.
        supplier_name: Associated supplier name.
        country: Origin country code.
        source_type: Ingestion source classification.
        ocr_confidence: Confidence score if OCR-derived.
        suspicious: Prompt injection flag.
        counter: TokenCounter instance (defaults to WhitespaceTokenCounter).
        target_tokens: Desired token size per chunk.
        max_tokens: Strict ceiling including context header (default 480).

    Returns:
        List of Chunk objects strictly within max_tokens ceiling.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    token_counter = counter if counter is not None else WhitespaceTokenCounter()
    header = _build_context_header(product_name or product_id, category, section)
    header_tokens = token_counter.count(header) if header else 0

    # Ensure max allowed body tokens
    max_body_tokens = max(50, max_tokens - header_tokens - 2)

    # Decompose into atomic units (preserving tables)
    blocks = _split_into_blocks(clean_text)
    atomic_units: list[str] = []

    for block in blocks:
        # If it's a table, keep whole table as unit unless it strictly exceeds max_body_tokens
        if "|" in block and "\n" in block:
            if token_counter.count(block) <= max_body_tokens:
                atomic_units.append(block)
            else:
                # Table exceeds max: split by table rows
                table_lines = [l for l in block.splitlines() if l.strip()]
                atomic_units.extend(table_lines)
        else:
            # Normal text block: check size
            if token_counter.count(block) <= target_tokens:
                atomic_units.append(block)
            else:
                # Split by sentences
                sentences = _split_sentences(block)
                for s in sentences:
                    if token_counter.count(s) <= max_body_tokens:
                        atomic_units.append(s)
                    else:
                        # Sentence exceeds max: split words
                        words = s.split()
                        cur_words: list[str] = []
                        for w in words:
                            cand = " ".join(cur_words + [w])
                            if token_counter.count(cand) > max_body_tokens:
                                if cur_words:
                                    atomic_units.append(" ".join(cur_words))
                                cur_words = [w]
                            else:
                                cur_words.append(w)
                        if cur_words:
                            atomic_units.append(" ".join(cur_words))

    # Assemble units into chunks up to target_tokens
    chunks: list[Chunk] = []
    doc_clean = re.sub(r"[^\w\-.]", "_", document)
    prefix = product_id or doc_clean

    chunk_type_label: Literal["card", "spec", "prose", "ocr_page", "ocr_block", "fixed"] = (
        "ocr_page" if source_type in ("ocr", "ocr_vision") else "prose"
    )

    current_units: list[str] = []

    def make_chunk(body_str: str, chunk_index: int) -> Chunk:
        full_text = f"{header}\n{body_str}" if header else body_str
        tok_count = token_counter.count(full_text)

        # Hard safety truncation if token count exceeds max_tokens
        if tok_count > max_tokens:
            body_words = body_str.split()
            while body_words:
                cand_body = " ".join(body_words)
                cand_full = f"{header}\n{cand_body}" if header else cand_body
                if token_counter.count(cand_full) <= max_tokens:
                    body_str = cand_body
                    full_text = cand_full
                    tok_count = token_counter.count(full_text)
                    break
                body_words.pop()

        tag_parts: list[str] = []
        if page is not None:
            tag_parts.append(f"P{page:02d}")
        if source_type in ("ocr", "ocr_vision"):
            tag_parts.append("OCR")
        else:
            tag_parts.append("PROSE")
        tag = "_".join(tag_parts)
        chunk_id = f"{prefix}_{tag}_{chunk_index:03d}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|{document}|{chunk_id}"))

        return Chunk(
            chunk_id=chunk_id,
            point_id=point_id,
            text=full_text,
            display_text=body_str,
            chunk_type=chunk_type_label,
            product_id=product_id,
            product_name=product_name,
            category=category,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            country=country,
            document=document,
            page=page,
            section=section,
            source_type=source_type,
            ocr_confidence=ocr_confidence,
            token_count=tok_count,
            suspicious=suspicious,
        )

    chunk_idx = 0
    for unit in atomic_units:
        cand_units = current_units + [unit]
        cand_body = "\n\n".join(cand_units)
        cand_tokens = token_counter.count(f"{header}\n{cand_body}" if header else cand_body)

        if cand_tokens <= target_tokens:
            current_units.append(unit)
        elif cand_tokens <= max_tokens and len(current_units) == 0:
            # Single unit fits within max ceiling
            current_units.append(unit)
        else:
            # Commit current chunk
            if current_units:
                body_text = "\n\n".join(current_units)
                chunks.append(make_chunk(body_text, chunk_idx))
                chunk_idx += 1
                current_units = [unit]
            else:
                current_units = [unit]

    if current_units:
        body_text = "\n\n".join(current_units)
        chunks.append(make_chunk(body_text, chunk_idx))

    return chunks
