"""Unit tests for evaluation metrics with 100% hand-computed ground truth."""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import (
    QueryMetrics,
    aggregate,
    aggregate_by_category,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_zero_total_relevant(self) -> None:
        assert recall_at_k(covered_count=0, total_relevant=0) == 0.0
        assert recall_at_k(covered_count=2, total_relevant=0) == 0.0

    def test_partial_and_full_coverage(self) -> None:
        # 1 covered out of 2 = 0.5
        assert recall_at_k(covered_count=1, total_relevant=2) == 0.5
        # 3 covered out of 4 = 0.75
        assert recall_at_k(covered_count=3, total_relevant=4) == 0.75
        # 4 covered out of 4 = 1.0
        assert recall_at_k(covered_count=4, total_relevant=4) == 1.0

    def test_clamped_bounds(self) -> None:
        # Cannot exceed 1.0
        assert recall_at_k(covered_count=5, total_relevant=4) == 1.0


class TestHitRateAtK:
    def test_hit_rate(self) -> None:
        labels = [False, True, False, False, False]
        assert hit_rate_at_k(labels, k=1) == 0.0
        assert hit_rate_at_k(labels, k=2) == 1.0
        assert hit_rate_at_k(labels, k=5) == 1.0

    def test_all_false_and_zero_k(self) -> None:
        assert hit_rate_at_k([False, False], k=2) == 0.0
        assert hit_rate_at_k([True], k=0) == 0.0


class TestPrecisionAtK:
    def test_precision_computations(self) -> None:
        labels = [True, False, True, False, False]
        # At k=1: 1/1 = 1.0
        assert precision_at_k(labels, k=1) == 1.0
        # At k=2: 1/2 = 0.5
        assert precision_at_k(labels, k=2) == 0.5
        # At k=5: 2/5 = 0.4
        assert precision_at_k(labels, k=5) == 0.4

    def test_edge_cases(self) -> None:
        assert precision_at_k([], k=5) == 0.0
        assert precision_at_k([True, True], k=0) == 0.0


class TestReciprocalRank:
    def test_first_rank(self) -> None:
        assert reciprocal_rank([True, False, False]) == 1.0

    def test_second_rank(self) -> None:
        assert reciprocal_rank([False, True, False]) == 0.5

    def test_third_rank(self) -> None:
        assert reciprocal_rank([False, False, True]) == pytest.approx(1.0 / 3.0)

    def test_no_relevant_hits(self) -> None:
        assert reciprocal_rank([False, False, False]) == 0.0
        assert reciprocal_rank([]) == 0.0

    def test_cutoff_k(self) -> None:
        # Third item is true, but cutoff is 2 -> should return 0.0
        assert reciprocal_rank([False, False, True], k=2) == 0.0
        # Cutoff 3 includes it
        assert reciprocal_rank([False, False, True], k=3) == pytest.approx(1.0 / 3.0)


class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        # If all top items are relevant, NDCG is 1.0
        assert ndcg_at_k([True, True, True], k=3) == 1.0

    def test_no_relevant(self) -> None:
        assert ndcg_at_k([False, False], k=2) == 0.0
        assert ndcg_at_k([], k=5) == 0.0
        assert ndcg_at_k([True], k=0) == 0.0

    def test_hand_computed_ndcg(self) -> None:
        # labels = [False, True], k=2, ideal_total=1
        # DCG = 0/log2(2) + 1/log2(3) = 1 / log2(3) = 0.63092975
        # IDCG = 1/log2(2) = 1.0
        # nDCG = 0.63092975 / 1.0 = 0.63092975
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert ndcg_at_k([False, True], k=2, ideal_total=1) == pytest.approx(expected)

        # labels = [True, False, False], k=3, ideal_total=2
        # DCG = 1 / log2(2) = 1.0
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
        expected_2 = 1.0 / (1.0 + 1.0 / math.log2(3))
        assert ndcg_at_k([True, False, False], k=3, ideal_total=2) == pytest.approx(expected_2)


class TestAggregation:
    def test_aggregate_empty(self) -> None:
        agg = aggregate([])
        assert agg.count == 0
        assert agg.mean_recall_at_k == 0.0

    def test_aggregate_answerable_filter(self) -> None:
        q1 = QueryMetrics(
            query_id="q1",
            category="direct_factual",
            split="dev",
            answerable=True,
            recall_at_k=1.0,
            mrr=1.0,
            hit_rate=1.0,
            precision_at_k=0.5,
            ndcg_at_k=1.0,
        )
        q2 = QueryMetrics(
            query_id="q2",
            category="direct_factual",
            split="dev",
            answerable=True,
            recall_at_k=0.5,
            mrr=0.5,
            hit_rate=1.0,
            precision_at_k=0.25,
            ndcg_at_k=0.6,
        )
        q_unans = QueryMetrics(
            query_id="q3",
            category="unanswerable_offtopic",
            split="dev",
            answerable=False,
            recall_at_k=0.0,
            mrr=0.0,
            hit_rate=0.0,
            precision_at_k=0.0,
            ndcg_at_k=0.0,
        )

        agg = aggregate([q1, q2, q_unans], filter_answerable=True)
        assert agg.count == 2
        assert agg.mean_recall_at_k == pytest.approx(0.75)
        assert agg.mrr == pytest.approx(0.75)
        assert agg.hit_rate == pytest.approx(1.0)
        assert agg.mean_precision_at_k == pytest.approx(0.375)
        assert agg.ndcg_at_k == pytest.approx(0.8)

    def test_aggregate_by_category(self) -> None:
        q1 = QueryMetrics("q1", "direct_factual", "dev", True, 1.0, 1.0, 1.0, 0.5, 1.0)
        q2 = QueryMetrics("q2", "ocr_only", "dev", True, 0.5, 0.5, 1.0, 0.25, 0.6)

        breakdown = aggregate_by_category([q1, q2])
        assert "direct_factual" in breakdown
        assert "ocr_only" in breakdown
        assert breakdown["direct_factual"].count == 1
        assert breakdown["ocr_only"].mean_recall_at_k == 0.5
