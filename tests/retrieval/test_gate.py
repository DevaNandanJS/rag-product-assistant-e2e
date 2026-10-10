"""Unit tests for threshold gating."""

from __future__ import annotations

from app.retrieval.gate import evaluate_gate


class TestSimilarityGate:
    def test_gate_passes_above_threshold(self) -> None:
        res = evaluate_gate(top_score=0.65, threshold=0.45, signal="dense_cosine", enabled=True)
        assert res.passed is True
        assert res.score == 0.65
        assert res.threshold == 0.45
        assert res.signal == "dense_cosine"

    def test_gate_fails_below_threshold(self) -> None:
        res = evaluate_gate(top_score=0.25, threshold=0.45, signal="dense_cosine", enabled=True)
        assert res.passed is False
        assert "below threshold" in res.reason

    def test_gate_disabled_always_passes(self) -> None:
        res = evaluate_gate(top_score=0.05, threshold=0.45, signal="dense_cosine", enabled=False)
        assert res.passed is True
        assert "Gate disabled" in res.reason
