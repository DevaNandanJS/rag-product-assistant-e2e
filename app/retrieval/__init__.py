"""Retrieval components: embedders, vector store, fusion, reranking, entities, gating, and pipeline.
"""

from app.retrieval.embedder import (
    Embedder,
    FakeEmbedder,
    FakeSparseEmbedder,
    FastEmbedEmbedder,
    FastEmbedSparseEmbedder,
    SparseEmbedder,
)
from app.retrieval.entities import (
    apply_entity_boost,
    ensure_card_coverage,
    extract_entities,
    is_comparison_query,
)
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.gate import GateResult, evaluate_gate
from app.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from app.retrieval.rerank import CrossEncoderReranker, FakeReranker, Reranker
from app.retrieval.store import QdrantStore

__all__ = [
    "CrossEncoderReranker",
    "Embedder",
    "FakeEmbedder",
    "FakeReranker",
    "FakeSparseEmbedder",
    "FastEmbedEmbedder",
    "FastEmbedSparseEmbedder",
    "GateResult",
    "QdrantStore",
    "Reranker",
    "RetrievalPipeline",
    "RetrievalResult",
    "SparseEmbedder",
    "apply_entity_boost",
    "ensure_card_coverage",
    "evaluate_gate",
    "extract_entities",
    "is_comparison_query",
    "reciprocal_rank_fusion",
]
