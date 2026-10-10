"""End-to-end retrieval pipeline coordinating dense search, BM25 sparse search,
RRF fusion, entity boost, cross-encoder reranking, and threshold gating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from qdrant_client.models import Filter

from app.core.config import Settings
from app.core.schemas import Chunk
from app.retrieval.embedder import Embedder
from app.retrieval.entities import (
    apply_entity_boost,
    ensure_card_coverage,
    extract_entities,
    is_comparison_query,
)
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.gate import GateResult, evaluate_gate
from app.retrieval.rerank import Reranker
from app.retrieval.store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Consolidated outcome of the retrieval pipeline execution."""

    chunks: list[Chunk]
    chunk_scores: list[tuple[Chunk, float]]
    gate: GateResult
    entities: list[str] = field(default_factory=list)
    is_comparison: bool = False
    debug_info: dict[str, Any] = field(default_factory=dict)


class RetrievalPipeline:
    """Coordinates hybrid retrieval, entity resolution, reranking, and threshold gating."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: QdrantStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.reranker = reranker

    def retrieve(
        self,
        question: str,
        filters: Filter | dict[str, Any] | None = None,
        chunker: str | None = None,
        top_k: int | None = None,
        candidates: int | None = None,
        retrieval_mode: str | None = None,
        gate_enabled: bool | None = None,
        threshold: float | None = None,
        entity_boost: bool = True,
    ) -> RetrievalResult:
        """Execute end-to-end retrieval for a question.

        Args:
            question: User query string.
            filters: Optional Qdrant metadata filters.
            chunker: Chunker name or default.
            top_k: Number of final chunks to return (default settings.TOP_K).
            candidates: Number of candidate chunks to fetch (default settings.CANDIDATES).
            retrieval_mode: 'hybrid', 'dense', or 'sparse' (default settings.RETRIEVAL_MODE).
            gate_enabled: Whether to enforce similarity gating (default settings.GATE_ENABLED).
            threshold: Gating cutoff (default settings.SCORE_THRESHOLD).
            entity_boost: Whether to apply entity score boosting and card coverage.

        Returns:
            RetrievalResult containing selected chunks, gating verdict, and debug info.
        """
        active_chunker = chunker or self.settings.CHUNKER
        active_top_k = top_k if top_k is not None else self.settings.TOP_K
        active_candidates = candidates if candidates is not None else self.settings.CANDIDATES
        active_mode = (retrieval_mode or self.settings.RETRIEVAL_MODE).lower()
        active_gate = gate_enabled if gate_enabled is not None else self.settings.GATE_ENABLED
        active_threshold = threshold if threshold is not None else self.settings.SCORE_THRESHOLD

        # 1. Entity resolution & comparison query detection
        entities = extract_entities(question)
        is_comp = is_comparison_query(question, entities)

        dense_results: list[tuple[Chunk, float]] = []
        sparse_results: list[tuple[Chunk, float]] = []
        fused_candidates: list[tuple[Chunk, float]] = []

        gate_score = 0.0
        gate_signal = "dense_cosine"

        # 2. Execution according to retrieval mode
        if active_mode in ("dense", "hybrid"):
            query_vec = self.embedder.embed([question])[0]
            dense_results = self.store.dense_search_with_scores(
                query_vector=query_vec,
                top_k=active_candidates,
                chunker=active_chunker,
                filters=filters,
            )
            if dense_results:
                gate_score = dense_results[0][1]
                gate_signal = "dense_cosine"

        if active_mode in ("sparse", "hybrid"):
            sparse_results = self.store.sparse_search_with_scores(
                query_text=question,
                top_k=active_candidates,
                chunker=active_chunker,
                filters=filters,
            )
            if active_mode == "sparse" and sparse_results:
                gate_score = sparse_results[0][1]
                gate_signal = "sparse_bm25"

        if active_mode == "hybrid":
            fused_candidates = reciprocal_rank_fusion(
                [dense_results, sparse_results],
                k=self.settings.RRF_K,
            )
        elif active_mode == "dense":
            fused_candidates = dense_results
        else:  # sparse
            fused_candidates = sparse_results

        # 3. Entity boosting & Card coverage
        pool = [chunk for chunk, _ in fused_candidates]
        if entity_boost and entities:
            fused_candidates = apply_entity_boost(
                fused_candidates,
                entities,
                boost=self.settings.ENTITY_BOOST,
            )

        if entity_boost and is_comp and len(entities) >= 2:
            fused_candidates = ensure_card_coverage(
                fused_candidates,
                entities,
                all_chunks_pool=pool,
            )

        # 4. Cross-Encoder Reranking
        if self.reranker is not None and fused_candidates:
            chunks_to_rerank = [chunk for chunk, _ in fused_candidates[:active_candidates]]
            reranked_pairs = self.reranker.rerank(
                query=question,
                chunks=chunks_to_rerank,
                top_k=active_top_k,
            )
            final_pairs = reranked_pairs
            if reranked_pairs:
                gate_score = reranked_pairs[0][1]
                gate_signal = "cross_encoder"
        else:
            final_pairs = fused_candidates[:active_top_k]

        # 5. Threshold Gate Evaluation
        gate_result = evaluate_gate(
            top_score=gate_score,
            threshold=active_threshold,
            signal=gate_signal,
            enabled=active_gate,
        )

        debug_info = {
            "dense_ranks": [
                {"chunk_id": c.chunk_id, "score": s}
                for c, s in dense_results[:active_top_k]
            ],
            "sparse_ranks": [
                {"chunk_id": c.chunk_id, "score": s}
                for c, s in sparse_results[:active_top_k]
            ],
            "fused_ranks": [
                {"chunk_id": c.chunk_id, "score": s}
                for c, s in fused_candidates[:active_top_k]
            ],
        }

        return RetrievalResult(
            chunks=[c for c, _ in final_pairs],
            chunk_scores=final_pairs,
            gate=gate_result,
            entities=entities,
            is_comparison=is_comp,
            debug_info=debug_info,
        )
