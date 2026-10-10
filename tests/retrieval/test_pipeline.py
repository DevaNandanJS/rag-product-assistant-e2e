"""Integration unit tests for RetrievalPipeline using FakeEmbedder and in-memory Qdrant."""

from __future__ import annotations

import uuid

from app.core.config import Settings
from app.core.schemas import Chunk
from app.retrieval.embedder import FakeEmbedder
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.rerank import FakeReranker
from app.retrieval.store import QdrantStore


def _make_chunk(cid: str, pid: str, text: str, chunk_type: str = "card") -> Chunk:
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|catalog.json|{cid}"))
    return Chunk(
        chunk_id=cid,
        point_id=point_id,
        text=text,
        display_text=text,
        chunk_type=chunk_type,
        document="catalog.json",
        product_id=pid,
        source_type="structured",
        token_count=12,
    )


class TestRetrievalPipeline:
    def test_pipeline_hybrid_mode(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)

        chunks = [
            _make_chunk("C1", "PKG-120", "CartonPro 1200 Sealer throughput 18 ctn/min"),
            _make_chunk("C2", "PKG-220", "FlexWrap 220 Pallet Wrapper capacity 2000 kg"),
            _make_chunk("C3", "WHS-1800", "StackStore 1800 Pallet Rack Bay 1800 kg"),
        ]
        store.replace_document("catalog.json", chunks)

        pipeline = RetrievalPipeline(
            settings=test_settings,
            embedder=embedder,
            store=store,
            reranker=FakeReranker(),
        )

        res = pipeline.retrieve(
            question="What is the throughput of PKG-120?",
            retrieval_mode="hybrid",
            top_k=2,
            candidates=5,
            gate_enabled=False,
            entity_boost=True,
        )

        assert len(res.chunks) == 2
        # PKG-120 chunk should be present and boosted
        assert "PKG-120" in res.entities
        assert any(c.product_id == "PKG-120" for c in res.chunks)
        assert res.gate.passed is True
        assert "dense_ranks" in res.debug_info
        assert "sparse_ranks" in res.debug_info
        assert "fused_ranks" in res.debug_info

    def test_pipeline_dense_and_sparse_modes(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)

        chunks = [
            _make_chunk("C1", "REF-320", "FrostHarbor CF320 REF-320 Chest Freezer -18C to -24C"),
            _make_chunk("C2", "REF-700", "PolarVault VC700 REF-700 Display Chiller +2C to +8C"),
        ]
        store.replace_document("catalog.json", chunks)

        pipeline = RetrievalPipeline(
            settings=test_settings,
            embedder=embedder,
            store=store,
        )

        res_dense = pipeline.retrieve("Chest freezer", retrieval_mode="dense", top_k=1)
        assert len(res_dense.chunks) == 1

        res_sparse = pipeline.retrieve("REF-320", retrieval_mode="sparse", top_k=1)
        assert len(res_sparse.chunks) == 1
        assert res_sparse.chunks[0].chunk_id == "C1"
