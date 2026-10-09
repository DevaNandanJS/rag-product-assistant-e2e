"""Fixed-size sliding window chunker (Baseline B0).
500-token sliding window, 50-token overlap, zero context headers.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

from app.core.schemas import Chunk
from app.ingestion.chunkers.tokens import TokenCounter


def chunk_fixed(
    text: str,
    document: str,
    page: int | None,
    source_type: Literal["structured", "extracted", "ocr", "ocr_vision"],
    ocr_confidence: float | None,
    counter: TokenCounter,
    window: int = 500,
    overlap: int = 50,
    suspicious: bool = False,
) -> list[Chunk]:
    """Split text into fixed-size chunks with a sliding window.

    Args:
        text: Raw or cleaned text to chunk.
        document: Originating filename.
        page: Optional 1-indexed page number.
        source_type: Ingestion source classification.
        ocr_confidence: Confidence score if OCR-derived.
        counter: TokenCounter instance.
        window: Maximum tokens per chunk.
        overlap: Token overlap between adjacent chunks.
        suspicious: Injection / security flag.

    Returns:
        List of Chunk objects with chunk_type="fixed" and zero context headers.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    doc_clean = re.sub(r"[^\w\-.]", "_", document)
    words = clean_text.split()
    if not words:
        return []

    total_tokens = counter.count(clean_text)
    if total_tokens <= window:
        chunk_id = f"{doc_clean}_FIX_000"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|{document}|{chunk_id}"))
        return [
            Chunk(
                chunk_id=chunk_id,
                point_id=point_id,
                text=clean_text,
                display_text=clean_text,
                chunk_type="fixed",
                document=document,
                page=page,
                source_type=source_type,
                ocr_confidence=ocr_confidence,
                token_count=total_tokens,
                suspicious=suspicious,
            )
        ]

    chunks: list[Chunk] = []
    start_idx = 0
    chunk_idx = 0

    while start_idx < len(words):
        # Expand end_idx until window limit or end of words
        end_idx = start_idx + 1
        while end_idx < len(words):
            candidate = " ".join(words[start_idx : end_idx + 1])
            if counter.count(candidate) > window:
                break
            end_idx += 1

        chunk_words = words[start_idx:end_idx]
        chunk_str = " ".join(chunk_words)
        tok_count = counter.count(chunk_str)

        chunk_id = f"{doc_clean}_FIX_{chunk_idx:03d}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|{document}|{chunk_id}"))

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                point_id=point_id,
                text=chunk_str,
                display_text=chunk_str,
                chunk_type="fixed",
                document=document,
                page=page,
                source_type=source_type,
                ocr_confidence=ocr_confidence,
                token_count=tok_count,
                suspicious=suspicious,
            )
        )
        chunk_idx += 1

        if end_idx >= len(words):
            break

        # Calculate overlap backwards from end_idx
        overlap_idx = end_idx - 1
        while overlap_idx > start_idx:
            overlap_str = " ".join(words[overlap_idx:end_idx])
            if counter.count(overlap_str) >= overlap:
                break
            overlap_idx -= 1

        # Advance start_idx ensuring forward progress
        next_start = max(start_idx + 1, overlap_idx)
        start_idx = next_start

    return chunks
