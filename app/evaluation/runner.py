"""Evaluation harness runner for executing benchmark runs and scoring retrievals.

Also provides AnswerEvalRunner for full pipeline evaluation (generation + LLM-as-a-Judge).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
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


# ---------------------------------------------------------------------------
# Phase 9: Answer-level evaluation with GenerationService + LLM-as-a-Judge
# ---------------------------------------------------------------------------


@dataclass
class AnswerRecord:
    """Full per-question record: generated answer, sources, latency, and judge score."""

    question_id: str
    question: str
    split: str
    category: str
    answerable: bool
    reference_answer: str
    must_include: list[str]
    generated_answer: str
    context_used: str           # concatenated source passage text sent to LLM
    sources_count: int          # number of retrieved chunks used
    citation_count: int         # number of [S#] markers found in generated answer
    latency_ms: float
    ttft_ms: float
    gate_passed: bool
    judge_correctness: float | None = None
    judge_groundedness: float | None = None
    judge_unsupported_claims: list[str] = field(default_factory=list)
    judge_notes: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "split": self.split,
            "category": self.category,
            "answerable": self.answerable,
            "reference_answer": self.reference_answer,
            "must_include": self.must_include,
            "generated_answer": self.generated_answer,
            "context_used": self.context_used[:500],  # truncate in output for readability
            "sources_count": self.sources_count,
            "citation_count": self.citation_count,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "gate_passed": self.gate_passed,
            "judge_correctness": self.judge_correctness,
            "judge_groundedness": self.judge_groundedness,
            "judge_unsupported_claims": self.judge_unsupported_claims,
            "judge_notes": self.judge_notes,
            "error": self.error,
        }


@dataclass
class AnswerAggregateMetrics:
    """Aggregated answer-level evaluation metrics across a split."""

    split: str
    total_questions: int
    answerable_count: int
    unanswerable_count: int
    # Gate metrics
    gate_pass_rate: float           # % of all questions that passed the gate
    correct_rejection_rate: float   # % of unanswerable questions correctly gated out
    false_rejection_rate: float     # % of answerable questions incorrectly gated out
    # Judge metrics (over questions that reached generation)
    mean_correctness: float
    mean_groundedness: float
    citation_precision: float       # avg cited_count / sources_count for answered questions
    # Latency
    median_latency_ms: float
    p95_latency_ms: float
    median_ttft_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "total_questions": self.total_questions,
            "answerable_count": self.answerable_count,
            "unanswerable_count": self.unanswerable_count,
            "gate_pass_rate": self.gate_pass_rate,
            "correct_rejection_rate": self.correct_rejection_rate,
            "false_rejection_rate": self.false_rejection_rate,
            "mean_correctness": self.mean_correctness,
            "mean_groundedness": self.mean_groundedness,
            "citation_precision": self.citation_precision,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "median_ttft_ms": self.median_ttft_ms,
        }


class AnswerEvalRunner:
    """Orchestrates full answer-level evaluation: retrieval → generation → judge.

    Responsibilities:
    - Load questions from eval/questions.json
    - Call GenerationService.generate() for each question (async)
    - Collect token output, sources, gate status, and latency from SSE events
    - Optionally call LLMJudge.score() for correctness + groundedness grading
    - Save per-question records to eval/results/answer_eval_{split}.jsonl
    - Compute and return aggregate metrics

    Single-responsibility: purely drives the answer evaluation pipeline.
    EvalRunner handles retrieval metrics; AnswerEvalRunner handles answer metrics.
    """

    def __init__(
        self,
        settings: Settings,
        generation_service: Any,  # GenerationService (avoid circular import)
        judge: Any | None = None,  # LLMJudge | None
    ) -> None:
        self._settings = settings
        self._service = generation_service
        self._judge = judge

    def load_questions(
        self,
        questions_path: str | Path = "eval/questions.json",
        split: str | None = None,
    ) -> list[EvalQuestion]:
        """Load and optionally filter questions by split."""
        path = Path(questions_path)
        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)
        questions = [EvalQuestion.from_dict(item) for item in raw_data]
        if split and split != "all":
            questions = [q for q in questions if q.split == split]
        return questions

    async def _generate_one(self, question: EvalQuestion) -> AnswerRecord:
        """Run generation pipeline for a single question and collect results."""
        from app.core.schemas import AskRequest

        t_start = time.perf_counter()
        tokens: list[str] = []
        sources_count = 0
        gate_passed = True
        ttft_ms = 0.0
        error: str | None = None
        context_parts: list[str] = []

        try:
            request = AskRequest(question=question.question, debug=False)
            async for event in self._service.generate(request):
                if event.type == "meta":
                    gate_info = event.payload.get("retrieval", {}).get("gate", {})
                    gate_passed = bool(gate_info.get("passed", True))
                elif event.type == "token":
                    content = event.payload.get("content", "")
                    if content and not tokens:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                    tokens.append(content)
                elif event.type == "sources":
                    srcs = event.payload.get("sources", [])
                    sources_count = len(srcs)
                    for s in srcs:
                        snippet = s.get("snippet", "")
                        if snippet:
                            context_parts.append(snippet)
        except Exception as exc:
            logger.error("[answer_eval] Error generating for %s: %s", question.id, exc)
            error = str(exc)

        latency_ms = (time.perf_counter() - t_start) * 1000
        generated_answer = "".join(tokens)
        context_used = " ".join(context_parts)

        # Count [S#] citation markers in the generated answer
        citation_count = len(set(re.findall(r"\[S\d+\]", generated_answer)))

        # Extract must_include from question (may not be in EvalQuestion dataclass)
        must_include: list[str] = []  # populated via raw data if available

        return AnswerRecord(
            question_id=question.id,
            question=question.question,
            split=question.split,
            category=question.category,
            answerable=question.answerable,
            reference_answer=question.reference_answer,
            must_include=must_include,
            generated_answer=generated_answer,
            context_used=context_used,
            sources_count=sources_count,
            citation_count=citation_count,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            gate_passed=gate_passed,
            error=error,
        )

    async def run_with_llm(
        self,
        questions_path: str | Path = "eval/questions.json",
        split: str = "dev",
        output_dir: str | Path = "eval/results",
        use_judge: bool = True,
    ) -> tuple[list[AnswerRecord], AnswerAggregateMetrics]:
        """Execute full answer evaluation for a split with optional judge scoring.

        Args:
            questions_path: Path to eval/questions.json.
            split: 'dev', 'holdout', or 'all'.
            output_dir: Directory to write answer_eval_{split}.jsonl results.
            use_judge: Whether to run LLM-as-a-Judge scoring.

        Returns:
            Tuple of (records list, aggregate metrics).
        """
        # Load raw JSON to get must_include field (not in EvalQuestion dataclass)
        path = Path(questions_path)
        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)
        must_include_map: dict[str, list[str]] = {
            item["id"]: item.get("must_include", []) for item in raw_data
        }

        questions = self.load_questions(questions_path, split=split)
        logger.info(
            "[answer_eval] Running answer eval on %d questions (split: %s)",
            len(questions),
            split,
        )

        # Generate answers sequentially (preserve API quota, avoid flooding)
        records: list[AnswerRecord] = []
        for i, q in enumerate(questions, 1):
            logger.info("[answer_eval] Generating %d/%d: %s", i, len(questions), q.id)
            print(f"  [{i}/{len(questions)}] {q.id}: {q.question[:60]}...")
            record = await self._generate_one(q)
            record.must_include = must_include_map.get(q.id, [])
            records.append(record)

        # Run judge scoring in batch (3 concurrent calls)
        if use_judge and self._judge is not None and self._judge.is_available:
            print(f"\nRunning LLM-as-a-Judge scoring on {len(records)} answers...")
            judge_items = [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "answerable": r.answerable,
                    "reference_answer": r.reference_answer,
                    "must_include": r.must_include,
                    "context": r.context_used,
                    "answer": r.generated_answer,
                }
                for r in records
            ]
            judge_scores = await self._judge.score_batch(judge_items, concurrency=3)
            score_map = {s.question_id: s for s in judge_scores}
            for record in records:
                js = score_map.get(record.question_id)
                if js:
                    if js.error:
                        logger.warning("[judge] Error for %s: %s", record.question_id, js.error)
                    else:
                        record.judge_correctness = js.correctness
                        record.judge_groundedness = js.groundedness
                        record.judge_unsupported_claims = js.unsupported_claims
                        record.judge_notes = js.notes
        elif use_judge and (self._judge is None or not self._judge.is_available):
            print("[WARN] Judge not available (GROQ_API_KEY not set); skipping judge scoring.")

        # Save JSONL results
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"answer_eval_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
        print(f"\n[INFO] Answer eval records saved to '{out_path}'")

        # Compute aggregate metrics
        metrics = _compute_answer_metrics(records, split)
        return records, metrics


def _compute_answer_metrics(
    records: list[AnswerRecord], split: str
) -> AnswerAggregateMetrics:
    """Compute aggregated answer-level evaluation metrics from a list of records."""
    import statistics

    total = len(records)
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]

    # Gate metrics
    answered = [r for r in records if r.gate_passed]
    gated_out = [r for r in records if not r.gate_passed]

    gate_pass_rate = len(answered) / total if total else 0.0
    correct_rejections = [r for r in unanswerable if not r.gate_passed]
    false_rejections = [r for r in answerable if not r.gate_passed]
    correct_rejection_rate = (
        len(correct_rejections) / len(unanswerable) if unanswerable else 0.0
    )
    false_rejection_rate = (
        len(false_rejections) / len(answerable) if answerable else 0.0
    )

    # Judge metrics — only for records that have judge scores
    judged = [r for r in records if r.judge_correctness is not None]
    mean_correctness = (
        statistics.mean(r.judge_correctness for r in judged) if judged else 0.0
    )
    mean_groundedness = (
        statistics.mean(r.judge_groundedness for r in judged) if judged else 0.0
    )

    # Citation precision: avg(citation_count / sources_count) for answered questions
    citation_ratios = [
        r.citation_count / r.sources_count
        for r in answered
        if r.sources_count > 0
    ]
    citation_precision = statistics.mean(citation_ratios) if citation_ratios else 0.0

    # Latency percentiles across all records
    latencies = [r.latency_ms for r in records if r.latency_ms > 0]
    ttfts = [r.ttft_ms for r in records if r.ttft_ms > 0]
    median_latency = statistics.median(latencies) if latencies else 0.0
    p95_latency = (
        sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else median_latency
    )
    median_ttft = statistics.median(ttfts) if ttfts else 0.0

    return AnswerAggregateMetrics(
        split=split,
        total_questions=total,
        answerable_count=len(answerable),
        unanswerable_count=len(unanswerable),
        gate_pass_rate=round(gate_pass_rate, 4),
        correct_rejection_rate=round(correct_rejection_rate, 4),
        false_rejection_rate=round(false_rejection_rate, 4),
        mean_correctness=round(mean_correctness, 4),
        mean_groundedness=round(mean_groundedness, 4),
        citation_precision=round(citation_precision, 4),
        median_latency_ms=round(median_latency, 1),
        p95_latency_ms=round(p95_latency, 1),
        median_ttft_ms=round(median_ttft, 1),
    )
