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
