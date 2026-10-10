"""Unit tests for rerankers."""

from __future__ import annotations

import uuid

from app.core.schemas import Chunk
from app.retrieval.rerank import FakeReranker


def _make_chunk(cid: str, text: str) -> Chunk:
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|doc.json|{cid}"))
    return Chunk(
        chunk_id=cid,
        point_id=point_id,
        text=text,
        display_text=text,
        chunk_type="spec",
        document="doc.json",
        source_type="structured",
        token_count=10,
    )


class TestReranker:
    def test_fake_reranker_reorders_by_overlap(self) -> None:
        c1 = _make_chunk("C1", "Conveyor belt for shipping packages")
        c2 = _make_chunk("C2", "Throughput 18 cartons per minute for carton sealer")

        reranker = FakeReranker()
        results = reranker.rerank(
            query="carton sealer throughput",
            chunks=[c1, c2],
            top_k=2,
        )

        assert len(results) == 2
        assert results[0][0].chunk_id == "C2"
        assert results[1][0].chunk_id == "C1"

    def test_fake_reranker_top_k_slice(self) -> None:
        chunks = [_make_chunk(f"C{i}", f"Text {i}") for i in range(10)]
        reranker = FakeReranker()
        results = reranker.rerank("Query", chunks, top_k=3)
        assert len(results) == 3

    def test_fake_reranker_empty(self) -> None:
        reranker = FakeReranker()
        assert reranker.rerank("Query", [], top_k=5) == []
