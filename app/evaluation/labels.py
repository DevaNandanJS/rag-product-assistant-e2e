"""Evidence-based labeling engine for retrieval evaluation.

Matches retrieved chunks against canonical manifest evidence (document, page,
and key factual assertions) without coupling to ephemeral chunk IDs.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from app.core.schemas import Chunk


@dataclass(frozen=True)
class EvidenceItem:
    """Canonical ground-truth evidence item for an evaluation question."""

    document: str
    page: int | None
    key_fact: str


def normalize_text(text: str) -> str:
    """Normalize text for invariant comparison.

    - Normalize degree symbols and variants
    - Unicode NFKC normalization
    - Lowercase
    - Standardize dashes (en-dash, em-dash, minus to hyphen)
    - Standardize multiplication sign (× to x)
    - Collapse multiple whitespace characters
    """
    if not text:
        return ""

    # Replace degree variants before NFKC decomposes ordinal indicator º to 'o'
    text = re.sub(r"[℃]", "°c", text)
    text = re.sub(r"[℉]", "°f", text)
    text = re.sub(r"[º]", "°", text)

    # NFKC normalizes compatibility characters
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Normalize dashes and symbols
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"[×]", "x", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_chunk_match(
    chunk: Chunk,
    evidence: EvidenceItem,
    fuzzy_threshold: float = 90.0,
) -> bool:
    """Determine whether a single chunk satisfies the specified evidence item.

    Rules:
    1. Document basename must match (case-insensitive).
    2. If evidence specifies a page, and the chunk has a page, they must match.
    3. The key_fact must appear in the chunk text either as a normalized substring
       or with RapidFuzz partial_ratio >= fuzzy_threshold.
    """
    # 1. Document match (compare basenames)
    chunk_doc = os.path.basename(chunk.document).lower()
    evidence_doc = os.path.basename(evidence.document).lower()
    if chunk_doc != evidence_doc:
        return False

    # 2. Page match (if both specify page)
    if evidence.page is not None and chunk.page is not None:
        if evidence.page != chunk.page:
            return False

    # 3. Key fact match
    norm_fact = normalize_text(evidence.key_fact)
    if not norm_fact:
        return True

    # Search in both chunk.text and chunk.display_text
    combined_chunk_text = f"{chunk.text} {chunk.display_text}"
    norm_chunk = normalize_text(combined_chunk_text)

    # Substring check
    if norm_fact in norm_chunk:
        return True

    # Fuzzy match fallback (e.g. for noisy OCR transcripts)
    score = fuzz.partial_ratio(norm_fact, norm_chunk)
    return score >= fuzzy_threshold


def label_retrieval(
    retrieved_chunks: Sequence[Chunk],
    evidence_items: Sequence[EvidenceItem],
    fuzzy_threshold: float = 90.0,
) -> list[bool]:
    """Label retrieved chunks as relevant (True) or irrelevant (False).

    Returns a boolean list of length len(retrieved_chunks).
    """
    if not evidence_items:
        return [False] * len(retrieved_chunks)

    labels: list[bool] = []
    for chunk in retrieved_chunks:
        is_relevant = any(
            is_chunk_match(chunk, evidence, fuzzy_threshold=fuzzy_threshold)
            for evidence in evidence_items
        )
        labels.append(is_relevant)
    return labels


def covered_evidence_indices(
    retrieved_chunks: Sequence[Chunk],
    evidence_items: Sequence[EvidenceItem],
    fuzzy_threshold: float = 90.0,
) -> set[int]:
    """Return indices of all evidence items satisfied by at least one retrieved chunk."""
    covered: set[int] = set()
    for idx, evidence in enumerate(evidence_items):
        for chunk in retrieved_chunks:
            if is_chunk_match(chunk, evidence, fuzzy_threshold=fuzzy_threshold):
                covered.add(idx)
                break
    return covered
