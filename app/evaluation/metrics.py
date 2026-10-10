"""Evaluation metrics computation: Recall@K, MRR, Hit Rate@K, Precision@K, and nDCG@K.

Designed with 100% testable pure functions and typed dataclass outputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryMetrics:
    """Retrieval evaluation metrics for a single query."""

    query_id: str
    category: str
    split: str
    answerable: bool
    recall_at_k: float
    mrr: float
    hit_rate: float
    precision_at_k: float
    ndcg_at_k: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Macro-averaged retrieval evaluation metrics over a set of queries."""

    mean_recall_at_k: float
    mrr: float
    hit_rate: float
    mean_precision_at_k: float
    ndcg_at_k: float
    count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_recall_at_k": round(self.mean_recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mean_precision_at_k": round(self.mean_precision_at_k, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "count": self.count,
        }


def recall_at_k(covered_count: int, total_relevant: int) -> float:
    """Compute Recall@K given the count of distinct covered ground-truth items.

    Args:
        covered_count: Number of unique relevant items retrieved in top-k.
        total_relevant: Total number of unique relevant items required.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if total_relevant <= 0.
    """
    if total_relevant <= 0:
        return 0.0
    return min(1.0, max(0.0, covered_count / total_relevant))


def hit_rate_at_k(labels: Sequence[bool], k: int) -> float:
    """Compute Hit Rate@K (1.0 if any item in top-k is relevant, else 0.0)."""
    if k <= 0:
        return 0.0
    return 1.0 if any(labels[:k]) else 0.0


def precision_at_k(labels: Sequence[bool], k: int) -> float:
    """Compute Precision@K: relevant items in top-k divided by k.

    Args:
        labels: Relevance booleans for retrieved items in rank order.
        k: Cutoff rank.

    Returns:
        Float in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    top = labels[:k]
    return sum(1.0 for r in top if r) / float(k)


def reciprocal_rank(labels: Sequence[bool], k: int | None = None) -> float:
    """Compute Reciprocal Rank: 1 / (rank of first relevant item).

    Args:
        labels: Relevance booleans for retrieved items in rank order.
        k: Optional rank cutoff.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if no relevant item is found.
    """
    cutoff = len(labels) if k is None or k <= 0 else min(k, len(labels))
    for rank, is_rel in enumerate(labels[:cutoff], start=1):
        if is_rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels: Sequence[bool], k: int, ideal_total: int | None = None) -> float:
    """Compute Normalized Discounted Cumulative Gain at cutoff k (binary relevance).

    Args:
        labels: Relevance booleans for retrieved items in rank order.
        k: Cutoff rank.
        ideal_total: Total number of relevant items in the universe. If None,
            inferred from the count of true labels in `labels`.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if no relevant items exist in ideal list.
    """
    if k <= 0:
        return 0.0

    top = labels[:k]
    # DCG = sum( rel_i / log2(i + 1) ) for i starting at 1
    dcg = 0.0
    for rank, is_rel in enumerate(top, start=1):
        if is_rel:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal ranking: all 1s first
    total_rel = ideal_total if ideal_total is not None else sum(1 for r in labels if r)
    ideal_hits = min(k, total_rel)
    if ideal_hits <= 0:
        return 0.0

    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)

    return dcg / idcg if idcg > 0.0 else 0.0


def aggregate(
    query_metrics: Sequence[QueryMetrics],
    filter_answerable: bool = True,
) -> AggregateMetrics:
    """Compute macro-averaged retrieval metrics across queries.

    Args:
        query_metrics: Sequence of per-query evaluation metrics.
        filter_answerable: If True, aggregates only over queries marked answerable=True.

    Returns:
        AggregateMetrics instance.
    """
    selected = [
        qm for qm in query_metrics if (not filter_answerable or qm.answerable)
    ]
    if not selected:
        return AggregateMetrics(
            mean_recall_at_k=0.0,
            mrr=0.0,
            hit_rate=0.0,
            mean_precision_at_k=0.0,
            ndcg_at_k=0.0,
            count=0,
        )

    n = float(len(selected))
    return AggregateMetrics(
        mean_recall_at_k=sum(qm.recall_at_k for qm in selected) / n,
        mrr=sum(qm.mrr for qm in selected) / n,
        hit_rate=sum(qm.hit_rate for qm in selected) / n,
        mean_precision_at_k=sum(qm.precision_at_k for qm in selected) / n,
        ndcg_at_k=sum(qm.ndcg_at_k for qm in selected) / n,
        count=len(selected),
    )


def aggregate_by_category(
    query_metrics: Sequence[QueryMetrics],
) -> dict[str, AggregateMetrics]:
    """Breakdown aggregate metrics by query category."""
    categories: dict[str, list[QueryMetrics]] = {}
    for qm in query_metrics:
        categories.setdefault(qm.category, []).append(qm)

    breakdown: dict[str, AggregateMetrics] = {}
    for cat, items in categories.items():
        # Include all questions for category aggregation (e.g. unanswerable queries)
        breakdown[cat] = aggregate(items, filter_answerable=False)
    return breakdown
