"""Similarity and confidence score threshold gating for filtering off-topic/unanswerable queries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    """Outcome of threshold gate evaluation."""

    passed: bool
    score: float
    threshold: float
    signal: str  # e.g., "dense_cosine" or "cross_encoder"
    reason: str = ""


def evaluate_gate(
    top_score: float,
    threshold: float = 0.45,
    signal: str = "dense_cosine",
    enabled: bool = True,
) -> GateResult:
    """Evaluate whether top candidate relevance score meets acceptance threshold.

    IMPORTANT: Per ADR-07 and Section 8, RRF rank scores CANNOT be used as a gating signal
    because RRF is rank-based and assigns top rank scores even to entirely irrelevant items.
    The gating signal must be absolute dense cosine similarity or cross-encoder logits.

    Args:
        top_score: The absolute similarity or cross-encoder score of top retrieved candidate.
        threshold: Score cutoff for acceptance (default 0.45).
        signal: Indicator of metric origin ("dense_cosine" or "cross_encoder").
        enabled: If False, gate always passes.

    Returns:
        GateResult indicating whether retrieval passes the threshold gate.
    """
    if not enabled:
        return GateResult(
            passed=True,
            score=top_score,
            threshold=threshold,
            signal=signal,
            reason="Gate disabled by configuration.",
        )

    passed = top_score >= threshold
    if passed:
        reason = f"Score {top_score:.4f} meets threshold {threshold:.4f}."
    else:
        reason = (
            f"Score {top_score:.4f} is below threshold {threshold:.4f}; "
            "query is unanswerable/off-topic."
        )

    return GateResult(
        passed=passed,
        score=top_score,
        threshold=threshold,
        signal=signal,
        reason=reason,
    )
