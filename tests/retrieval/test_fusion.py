"""Unit tests for Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import uuid

import pytest

from app.core.schemas import Chunk
from app.retrieval.fusion import reciprocal_rank_fusion


def _make_chunk(cid: str) -> Chunk:
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|doc.json|{cid}"))
    return Chunk(
        chunk_id=cid,
        point_id=point_id,
        text=f"Text for {cid}",
        display_text=f"Text for {cid}",
        chunk_type="card",
        document="doc.json",
        source_type="structured",
        token_count=10,
    )


class TestReciprocalRankFusion:
    def test_hand_computed_rrf(self) -> None:
        c1 = _make_chunk("C1")
        c2 = _make_chunk("C2")
        c3 = _make_chunk("C3")

        # List 1: C1 (rank 1), C2 (rank 2), C3 (rank 3)
        list1 = [(c1, 0.9), (c2, 0.8), (c3, 0.7)]
        # List 2: C2 (rank 1), C1 (rank 2)
        list2 = [(c2, 10.0), (c1, 5.0)]

        # k = 60
        # C1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639344 + 0.01612903 = 0.03252247
        # C2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.03252247
        # C3: 1/(60+3) = 1/63 = 0.01587301
        fused = reciprocal_rank_fusion([list1, list2], k=60)

        assert len(fused) == 3
        c_map = {c.chunk_id: score for c, score in fused}

        expected_c1_c2 = (1.0 / 61) + (1.0 / 62)
        expected_c3 = 1.0 / 63

        assert pytest.approx(c_map["C1"], rel=1e-5) == expected_c1_c2
        assert pytest.approx(c_map["C2"], rel=1e-5) == expected_c1_c2
        assert pytest.approx(c_map["C3"], rel=1e-5) == expected_c3

        # Both C1 and C2 score higher than C3
        assert c_map["C1"] > c_map["C3"]
        assert c_map["C2"] > c_map["C3"]

    def test_empty_lists(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_disjoint_lists(self) -> None:
        c1 = _make_chunk("C1")
        c2 = _make_chunk("C2")

        list1 = [(c1, 1.0)]
        list2 = [(c2, 1.0)]

        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert len(fused) == 2
        # Both appear at rank 1 in their respective lists, so equal RRF score
        assert fused[0][1] == fused[1][1]
