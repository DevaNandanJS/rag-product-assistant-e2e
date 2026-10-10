"""Reciprocal Rank Fusion (RRF) algorithm for merging multiple ranked retrieval results.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.schemas import Chunk


def reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[tuple[Chunk, float]]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """Compute Reciprocal Rank Fusion scores across multiple ranked candidate lists.

    Formula:
        RRF_Score(d) = sum_{m in M} (1 / (k + r_m(d)))
    where r_m(d) is the 1-based rank position of chunk d in rank list m.

    Args:
        rank_lists: Sequence of ranked candidate lists, each containing (Chunk, score) tuples.
        k: Smoothing constant (default 60, standard in literature).

    Returns:
        List of (Chunk, rrf_score) tuples sorted in descending RRF score order.
    """
    chunk_map: dict[str, Chunk] = {}
    rrf_scores: dict[str, float] = {}

    for rank_list in rank_lists:
        for rank_idx, (chunk, _) in enumerate(rank_list, start=1):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk
                rrf_scores[cid] = 0.0

            rrf_scores[cid] += 1.0 / (k + rank_idx)

    # Sort descending by fused RRF score; break ties by chunk_id for determinism
    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )

    return [(chunk_map[cid], score) for cid, score in sorted_items]
