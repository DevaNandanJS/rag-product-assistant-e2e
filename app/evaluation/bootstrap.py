"""Paired bootstrap confidence interval and significance test for retrieval metrics.

Performs paired query resampling (with replacement) to compute empirical 95%
confidence intervals on metric deltas (improved - baseline).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a paired bootstrap significance test."""

    baseline_mean: float
    improved_mean: float
    delta_mean: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "baseline_mean": round(self.baseline_mean, 4),
            "improved_mean": round(self.improved_mean, 4),
            "delta_mean": round(self.delta_mean, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "p_value": round(self.p_value, 4),
            "significant": self.significant,
        }


def paired_bootstrap(
    baseline_scores: Sequence[float],
    improved_scores: Sequence[float],
    n_iterations: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> BootstrapResult:
    """Perform paired bootstrap resampling to calculate confidence intervals on delta.

    Args:
        baseline_scores: Sequence of per-query metric values for the baseline system.
        improved_scores: Sequence of per-query metric values for the evaluated system.
        n_iterations: Number of bootstrap iterations (default: 1000).
        seed: Random seed for deterministic reproducibility (default: 42).
        ci_level: Confidence interval span, e.g. 0.95 for 95% CI.

    Returns:
        BootstrapResult with mean delta, empirical CI bounds, and p-value.
    """
    if len(baseline_scores) != len(improved_scores):
        raise ValueError(
            f"Length mismatch: baseline has {len(baseline_scores)} items, "
            f"improved has {len(improved_scores)} items."
        )

    n = len(baseline_scores)
    if n == 0:
        return BootstrapResult(
            baseline_mean=0.0,
            improved_mean=0.0,
            delta_mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            significant=False,
        )

    rng = random.Random(seed)
    deltas: list[float] = []
    base_means: list[float] = []
    imp_means: list[float] = []

    for _ in range(n_iterations):
        sample_indices = [rng.randint(0, n - 1) for _ in range(n)]
        b_mean = sum(baseline_scores[i] for i in sample_indices) / float(n)
        i_mean = sum(improved_scores[i] for i in sample_indices) / float(n)
        base_means.append(b_mean)
        imp_means.append(i_mean)
        deltas.append(i_mean - b_mean)

    deltas.sort()
    alpha = (1.0 - ci_level) / 2.0
    lower_idx = int(alpha * n_iterations)
    upper_idx = int((1.0 - alpha) * n_iterations)
    upper_idx = min(upper_idx, n_iterations - 1)

    ci_lower = deltas[lower_idx]
    ci_upper = deltas[upper_idx]

    # One-sided empirical p-value: fraction of iterations where improved did not beat baseline
    non_positive = sum(1 for d in deltas if d <= 0.0)
    p_value = non_positive / float(n_iterations)

    actual_base_mean = sum(baseline_scores) / float(n)
    actual_imp_mean = sum(improved_scores) / float(n)
    delta_mean = actual_imp_mean - actual_base_mean

    # Significant at alpha if lower CI bound > 0 (for improvement)
    significant = ci_lower > 0.0

    return BootstrapResult(
        baseline_mean=actual_base_mean,
        improved_mean=actual_imp_mean,
        delta_mean=delta_mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        significant=significant,
    )


def bootstrap_ci(
    scores: Sequence[float],
    n_iterations: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Calculate single-distribution bootstrap confidence interval: (mean, ci_lower, ci_upper)."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_iterations):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / float(n))

    means.sort()
    alpha = (1.0 - ci_level) / 2.0
    lower_idx = int(alpha * n_iterations)
    upper_idx = int((1.0 - alpha) * n_iterations)
    upper_idx = min(upper_idx, n_iterations - 1)

    actual_mean = sum(scores) / float(n)
    return actual_mean, means[lower_idx], means[upper_idx]
