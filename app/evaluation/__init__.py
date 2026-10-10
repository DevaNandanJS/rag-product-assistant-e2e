"""Evaluation package: Ground truth labels, metrics, runner, bootstrap, and reports."""

from app.evaluation.bootstrap import BootstrapResult, paired_bootstrap
from app.evaluation.labels import EvidenceItem, is_chunk_match, label_retrieval
from app.evaluation.metrics import (
    AggregateMetrics,
    QueryMetrics,
    aggregate,
    aggregate_by_category,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.report import (
    format_comparison_table,
    format_markdown_table,
    save_result,
)
from app.evaluation.runner import EvalConfig, EvalQuestion, EvalResult, EvalRunner

__all__ = [
    "EvidenceItem",
    "is_chunk_match",
    "label_retrieval",
    "QueryMetrics",
    "AggregateMetrics",
    "recall_at_k",
    "hit_rate_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "aggregate",
    "aggregate_by_category",
    "paired_bootstrap",
    "BootstrapResult",
    "EvalConfig",
    "EvalQuestion",
    "EvalResult",
    "EvalRunner",
    "format_markdown_table",
    "format_comparison_table",
    "save_result",
]
