"""Evaluation harness runner for executing benchmark runs and scoring retrievals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.config import Settings
from app.evaluation.labels import (
    EvidenceItem,
    covered_evidence_indices,
    label_retrieval,
)
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
from app.retrieval.embedder import Embedder, FakeEmbedder
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.rerank import CrossEncoderReranker, FakeReranker
from app.retrieval.store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Benchmark evaluation configuration loaded from YAML."""

    run_id: str
    description: str = ""
    chunker: str = "fixed"
    retrieval_mode: str = "dense"
    top_k: int = 5
    candidates: int = 20
    gate_enabled: bool = False
    reranker: str = ""
    split: str = "dev"
    entity_boost: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class EvalQuestion:
    """Single benchmark evaluation question."""

    id: str
    split: str
    category: str
    question: str
    answerable: bool
    reference_answer: str
    evidence: list[EvidenceItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalQuestion:
        evidence_list = [
            EvidenceItem(
                document=ev.get("document", ""),
                page=ev.get("page"),
                key_fact=ev.get("key_fact", ""),
            )
            for ev in data.get("evidence", [])
        ]
        return cls(
            id=data["id"],
            split=data.get("split", "dev"),
            category=data.get("category", "direct_factual"),
            question=data["question"],
            answerable=data.get("answerable", True),
            reference_answer=data.get("reference_answer", ""),
            evidence=evidence_list,
        )


@dataclass
class EvalResult:
    """Complete results of an evaluation run."""

    run_id: str
    config: EvalConfig
    split: str
    overall: AggregateMetrics
    by_category: dict[str, AggregateMetrics]
    per_query: list[QueryMetrics]
    retrieved_chunk_ids: dict[str, list[str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": asdict(self.config),
            "split": self.split,
            "overall": self.overall.as_dict(),
            "by_category": {cat: m.as_dict() for cat, m in self.by_category.items()},
            "per_query": [asdict(qm) for qm in self.per_query],
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
        }


class EvalRunner:
    """Orchestrates retrieval evaluation against ground-truth questions."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: QdrantStore,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def load_questions(
        self,
        questions_path: str | Path = "eval/questions.json",
        split: str | None = None,
    ) -> list[EvalQuestion]:
        """Load and optionally filter questions by split ('dev', 'holdout', or None for all)."""
        path = Path(questions_path)
        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)

        questions = [EvalQuestion.from_dict(item) for item in raw_data]
        if split and split != "all":
            questions = [q for q in questions if q.split == split]
        return questions

    def run(
        self,
        config: EvalConfig,
        questions_path: str | Path = "eval/questions.json",
        split_override: str | None = None,
    ) -> EvalResult:
        """Execute the evaluation run specified by config.

        Args:
            config: EvalConfig configuration.
            questions_path: Path to questions.json file.
            split_override: If provided, overrides config.split ('dev', 'holdout', 'all').

        Returns:
            EvalResult containing summary metrics and per-query breakdown.
        """
        active_split = split_override or config.split
        questions = self.load_questions(questions_path, split=active_split)
        logger.info(
            "Running eval '%s' on %d questions (split: %s, chunker: %s, top_k: %d)",
            config.run_id,
            len(questions),
            active_split,
            config.chunker,
            config.top_k,
        )

        per_query_metrics: list[QueryMetrics] = []
        retrieved_chunk_ids: dict[str, list[str]] = {}

        reranker = None
        if config.reranker:
            if isinstance(self.embedder, FakeEmbedder):
                reranker = FakeReranker()
            else:
                reranker = CrossEncoderReranker(
                    model_name=config.reranker,
                    cache_dir=self.settings.FASTEMBED_CACHE_PATH,
                )

        pipeline = RetrievalPipeline(
            settings=self.settings,
            embedder=self.embedder,
            store=self.store,
            reranker=reranker,
        )

        for q in questions:
            if (
                config.retrieval_mode == "dense"
                and not config.reranker
                and not config.entity_boost
                and not config.gate_enabled
            ):
                # Baseline fast path
                query_vec = self.embedder.embed([q.question])[0]
                retrieved_chunks = self.store.dense_search(
                    query_vector=query_vec,
                    top_k=config.top_k,
                    chunker=config.chunker,
                )
            else:
                retrieval_res = pipeline.retrieve(
                    question=q.question,
                    chunker=config.chunker,
                    top_k=config.top_k,
                    candidates=config.candidates,
                    retrieval_mode=config.retrieval_mode,
                    gate_enabled=config.gate_enabled,
                    entity_boost=config.entity_boost,
                )
                if config.gate_enabled and not retrieval_res.gate.passed:
                    retrieved_chunks = []
                else:
                    retrieved_chunks = retrieval_res.chunks

            retrieved_chunk_ids[q.id] = [c.chunk_id for c in retrieved_chunks]

            labels = label_retrieval(retrieved_chunks, q.evidence)
            covered = covered_evidence_indices(retrieved_chunks, q.evidence)
            total_relevant = len(q.evidence)

            rec = recall_at_k(len(covered), total_relevant)
            rr = reciprocal_rank(labels, k=config.top_k)
            hr = hit_rate_at_k(labels, k=config.top_k)
            prec = precision_at_k(labels, k=config.top_k)
            ideal = total_relevant if total_relevant > 0 else None
            ndcg = ndcg_at_k(labels, k=config.top_k, ideal_total=ideal)

            qm = QueryMetrics(
                query_id=q.id,
                category=q.category,
                split=q.split,
                answerable=q.answerable,
                recall_at_k=rec,
                mrr=rr,
                hit_rate=hr,
                precision_at_k=prec,
                ndcg_at_k=ndcg,
            )
            per_query_metrics.append(qm)

        overall = aggregate(per_query_metrics, filter_answerable=True)
        by_cat = aggregate_by_category(per_query_metrics)

        return EvalResult(
            run_id=config.run_id,
            config=config,
            split=active_split,
            overall=overall,
            by_category=by_cat,
            per_query=per_query_metrics,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )
