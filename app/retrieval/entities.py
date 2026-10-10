"""Entity resolution, comparison query detection, and SKU boost scoring.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rapidfuzz import fuzz

from app.core.schemas import Chunk

# Mapping of canonical product IDs to their aliases and common keyword descriptions
PRODUCT_CATALOG: dict[str, list[str]] = {
    "PKG-120-PRO": [
        "PKG-120-PRO",
        "PKG120PRO",
        "CartonPro 1200 Pro",
        "CartonPro Pro",
        "1200 Pro",
    ],
    "PKG-120": [
        "PKG-120",
        "PKG120",
        "CartonPro 1200",
        "CartonPro",
        "carton sealer",
        "box sealer",
        "tape sealer",
    ],
    "PKG-220": [
        "PKG-220",
        "PKG220",
        "FlexWrap 220",
        "FlexWrap",
        "turntable pallet wrapper",
        "turntable wrapper",
        "pallet stretch wrapper",
        "pallet wrapper",
    ],
    "PKG-CF48": [
        "PKG-CF48",
        "PKGCF48",
        "CF-48",
        "CF48",
        "ClearFilm CF-48",
        "ClearFilm",
        "stretch film",
        "stretch wrap roll",
        "LLDPE stretch film",
    ],
    "WHS-1800": [
        "WHS-1800",
        "WHS1800",
        "StackStore 1800",
        "StackStore",
        "pallet rack bay",
        "pallet racking",
        "pallet rack",
        "teardrop rack",
    ],
    "WHS-1000": [
        "WHS-1000",
        "WHS1000",
        "LiftMate 1000",
        "LiftMate",
        "hydraulic stacker",
        "pallet stacker",
        "manual stacker",
        "hydraulic pallet stacker",
        "stacker",
    ],
    "WHS-400": [
        "WHS-400",
        "WHS400",
        "MoveEase 400",
        "MoveEase",
        "platform trolley",
        "steel trolley",
        "warehouse trolley",
        "trolley",
    ],
    "REF-320": [
        "REF-320",
        "REF320",
        "CF320",
        "FrostHarbor CF320",
        "FrostHarbor",
        "chest freezer",
        "commercial freezer",
        "freezer",
    ],
    "REF-700": [
        "REF-700",
        "REF700",
        "VC700",
        "PolarVault VC700",
        "PolarVault",
        "display chiller",
        "upright chiller",
        "glass door chiller",
        "commercial chiller",
        "chiller",
    ],
    "TEX-12": [
        "TEX-12",
        "TEX12",
        "ST-12",
        "ST12",
        "StitchPro ST-12",
        "StitchPro",
        "straight-stitch sewing machine",
        "industrial sewing machine",
        "lockstitch sewing machine",
        "sewing machine",
    ],
}

COMPARISON_KEYWORDS = (
    "compare",
    "comparison",
    "vs",
    "versus",
    "difference",
    "differ",
    "differences",
    "differ between",
    "between",
    "better",
    "choose between",
)


def extract_entities(question: str, threshold: float = 88.0) -> list[str]:
    """Extract product SKU entities from a natural language question.

    Uses a combination of exact regex pattern matches and RapidFuzz partial token matching
    against known product identifiers and aliases.

    Args:
        question: The user query or evaluation question.
        threshold: Fuzzy match score cutoff (0-100).

    Returns:
        Deduplicated list of matched product IDs in order of match/appearance.
    """
    q_norm = question.lower()
    matched_entities: list[str] = []

    # 1. Exact regex check for model numbers and hyphenated SKUs
    # Order patterns such that longer, specific variants (e.g. PKG-120-PRO) precede base variants
    sorted_catalog = sorted(PRODUCT_CATALOG.items(), key=lambda kv: len(kv[0]), reverse=True)

    for pid, aliases in sorted_catalog:
        # Check direct substring matching on normalized aliases
        for alias in aliases:
            a_norm = alias.lower()
            # If alias is short (e.g. "CF48", "ST12", "WHS-400"), enforce word boundary
            if len(a_norm) <= 7 or "-" in a_norm:
                pattern = r"(?:\b|_)" + re.escape(a_norm) + r"(?:\b|_)"
                if re.search(pattern, q_norm):
                    if pid not in matched_entities:
                        matched_entities.append(pid)
                    break
            else:
                # Direct phrase match
                if a_norm in q_norm:
                    if pid not in matched_entities:
                        matched_entities.append(pid)
                    break
                # Rapidfuzz partial ratio
                score = fuzz.partial_ratio(a_norm, q_norm)
                if score >= threshold:
                    if pid not in matched_entities:
                        matched_entities.append(pid)
                    break

    return matched_entities


def is_comparison_query(question: str, entities: list[str]) -> bool:
    """Detect whether a query is a cross-product comparison query.

    Args:
        question: The user query string.
        entities: List of resolved entity product IDs.

    Returns:
        True if query compares multiple products.
    """
    if len(entities) >= 2:
        return True

    q_lower = question.lower()
    for kw in COMPARISON_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, q_lower):
            return True

    return False


def apply_entity_boost(
    candidates: Sequence[tuple[Chunk, float]],
    entities: Sequence[str],
    boost: float = 1.25,
) -> list[tuple[Chunk, float]]:
    """Multiply retrieval score by boost factor for chunks matching identified entities.

    Args:
        candidates: Sequence of (Chunk, score) pairs.
        entities: List of identified product IDs.
        boost: Multiplier applied to matching chunks (default 1.25).

    Returns:
        List of (Chunk, boosted_score) re-sorted by boosted score descending.
    """
    if not entities or boost <= 1.0:
        return list(candidates)

    entity_set = set(entities)
    boosted: list[tuple[Chunk, float]] = []

    for chunk, score in candidates:
        if chunk.product_id and chunk.product_id in entity_set:
            new_score = score * boost
        else:
            new_score = score
        boosted.append((chunk, new_score))

    # Re-sort descending; break ties by chunk_id
    boosted.sort(key=lambda item: (item[1], item[0].chunk_id), reverse=True)
    return boosted


def ensure_card_coverage(
    candidates: list[tuple[Chunk, float]],
    entities: list[str],
    all_chunks_pool: list[Chunk],
) -> list[tuple[Chunk, float]]:
    """For comparison queries, ensure at least one Master Card Chunk is present for each entity.

    If an identified entity lacks a Master Card Chunk in the candidate list, this function
    injects the missing card chunk from the candidate pool.

    Args:
        candidates: Current top ranked (Chunk, score) tuples.
        entities: Identified comparison entities.
        all_chunks_pool: Available candidate chunks pool.

    Returns:
        Updated candidate list ensuring card coverage.
    """
    if len(entities) < 2:
        return candidates

    result = list(candidates)
    existing_cards = {
        c.product_id
        for c, _ in result
        if c.chunk_type == "card" and c.product_id
    }

    # Minimum score in result or fallback
    min_score = min((s for _, s in result), default=0.01)

    for ent in entities:
        if ent not in existing_cards:
            # Find a card chunk for this entity in all_chunks_pool
            for chunk in all_chunks_pool:
                if chunk.product_id == ent and chunk.chunk_type == "card":
                    # Check if this exact chunk is already in results with another type
                    if not any(c.chunk_id == chunk.chunk_id for c, _ in result):
                        result.append((chunk, min_score * 0.99))
                        existing_cards.add(ent)
                        break

    return result
