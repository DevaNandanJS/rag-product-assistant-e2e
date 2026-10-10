"""Unit tests for paired bootstrap confidence intervals and statistical significance."""

from __future__ import annotations

import pytest

from app.evaluation.bootstrap import bootstrap_ci, paired_bootstrap


class TestPairedBootstrap:
    def test_identical_distributions(self) -> None:
        scores = [0.8, 0.6, 0.4, 1.0, 0.2]
        res = paired_bootstrap(scores, scores, n_iterations=200, seed=42)
        assert res.delta_mean == pytest.approx(0.0)
        assert res.significant is False
        assert res.ci_lower <= 0.0 <= res.ci_upper

    def test_significant_improvement(self) -> None:
        baseline = [0.0, 0.1, 0.2, 0.0, 0.1]
        improved = [0.9, 0.8, 1.0, 0.9, 0.85]
        res = paired_bootstrap(baseline, improved, n_iterations=500, seed=42)
        assert res.delta_mean > 0.6
        assert res.ci_lower > 0.0
        assert res.significant is True
        assert res.p_value < 0.01

    def test_determinism(self) -> None:
        b = [0.2, 0.4, 0.6, 0.8]
        i = [0.4, 0.5, 0.7, 0.9]
        res1 = paired_bootstrap(b, i, n_iterations=300, seed=123)
        res2 = paired_bootstrap(b, i, n_iterations=300, seed=123)
        assert res1.ci_lower == res2.ci_lower
        assert res1.ci_upper == res2.ci_upper
        assert res1.p_value == res2.p_value

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            paired_bootstrap([1.0], [1.0, 2.0])

    def test_empty_scores(self) -> None:
        res = paired_bootstrap([], [])
        assert res.delta_mean == 0.0
        assert res.significant is False


class TestBootstrapCI:
    def test_single_distribution(self) -> None:
        scores = [1.0, 1.0, 1.0, 1.0]
        mean, lo, hi = bootstrap_ci(scores, n_iterations=100, seed=42)
        assert mean == 1.0
        assert lo == 1.0
        assert hi == 1.0

    def test_empty(self) -> None:
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
