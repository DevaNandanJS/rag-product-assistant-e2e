"""Unit tests for QdrantStore in offline in-memory mode."""

from __future__ import annotations

import uuid

from app.core.config import Settings
from app.core.schemas import Chunk
from app.retrieval.embedder import FakeEmbedder
from app.retrieval.store import QdrantStore


def _make_dummy_chunk(chunk_id: str, doc: str, text: str = "Test chunk body") -> Chunk:
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|{doc}|{chunk_id}"))
    return Chunk(
        chunk_id=chunk_id,
        point_id=point_id,
        text=text,
        display_text=text,
        chunk_type="card",
        document=doc,
        source_type="structured",
        token_count=10,
    )


class TestQdrantStore:
    """Verifies collection provisioning and document replacement in memory."""

    def test_ensure_collection_creates(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)
        col_name = store.collection_name()

        assert not store.collection_exists(col_name)
        store.ensure_collection()
        assert store.collection_exists(col_name)

    def test_replace_document_upserts_and_counts(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)

        chunks = [
            _make_dummy_chunk("C1", "doc1.json"),
            _make_dummy_chunk("C2", "doc1.json"),
            _make_dummy_chunk("C3", "doc1.json"),
        ]

        count = store.replace_document("doc1.json", chunks)
        assert count == 3
        assert store.count() == 3

    def test_replace_document_idempotent(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)

        chunks = [_make_dummy_chunk("C1", "doc1.json"), _make_dummy_chunk("C2", "doc1.json")]
        store.replace_document("doc1.json", chunks)
        assert store.count() == 2

        # Re-index identical document
        store.replace_document("doc1.json", chunks)
        assert store.count() == 2

    def test_replace_document_removes_orphans(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)

        chunks_v1 = [
            _make_dummy_chunk("C1", "doc1.json"),
            _make_dummy_chunk("C2", "doc1.json"),
            _make_dummy_chunk("C3", "doc1.json"),
        ]
        store.replace_document("doc1.json", chunks_v1)
        assert store.count() == 3

        # Update document with only 1 chunk (removing C2 and C3)
        chunks_v2 = [_make_dummy_chunk("C1", "doc1.json")]
        store.replace_document("doc1.json", chunks_v2)
        assert store.count() == 1

    def test_delete_collection(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)
        store.ensure_collection()
        col = store.collection_name()
        assert store.collection_exists(col)

        store.delete_collection()
        assert not store.collection_exists(col)

    def test_dense_search(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)

        chunks = [
            _make_dummy_chunk("C1", "doc1.json", "Throughput 18 ctn/min"),
            _make_dummy_chunk("C2", "doc1.json", "Voltage 230V AC"),
        ]
        store.replace_document("doc1.json", chunks)

        query_vec = embedder.embed(["throughput"])[0]
        hits = store.dense_search(query_vec, top_k=2)
        assert len(hits) == 2
        assert isinstance(hits[0], Chunk)

        hits_with_scores = store.dense_search_with_scores(query_vec, top_k=2)
        assert len(hits_with_scores) == 2
        assert isinstance(hits_with_scores[0][0], Chunk)
        assert isinstance(hits_with_scores[0][1], float)

    def test_sparse_search(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)

        chunks = [
            _make_dummy_chunk("C1", "doc1.json", "Throughput 18 ctn/min PKG-120"),
            _make_dummy_chunk("C2", "doc1.json", "Voltage 230V AC WHS-1800"),
        ]
        store.replace_document("doc1.json", chunks)

        hits = store.sparse_search("PKG-120", top_k=2)
        assert len(hits) >= 1
        assert hits[0].chunk_id == "C1"

        hits_with_scores = store.sparse_search_with_scores("PKG-120", top_k=2)
        assert len(hits_with_scores) >= 1
        assert hits_with_scores[0][0].chunk_id == "C1"
        assert hits_with_scores[0][1] > 0.0

    def test_metadata_filtering(self, test_settings: Settings) -> None:
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)

        chunk1 = _make_dummy_chunk("C1", "doc1.json", "Sealer specs")
        chunk1.product_id = "PKG-120"
        chunk1.category = "packaging_equipment"

        chunk2 = _make_dummy_chunk("C2", "doc2.json", "Racking specs")
        chunk2.product_id = "WHS-1800"
        chunk2.category = "warehouse_storage"

        store.replace_document("doc1.json", [chunk1])
        store.replace_document("doc2.json", [chunk2])

        query_vec = embedder.embed(["specs"])[0]
        # Filter for packaging_equipment only
        filtered_hits = store.dense_search(
            query_vec,
            top_k=5,
            filters={"category": "packaging_equipment"},
        )
        assert len(filtered_hits) == 1
        assert filtered_hits[0].chunk_id == "C1"

