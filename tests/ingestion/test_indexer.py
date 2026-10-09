"""Integration tests for Indexer with DocumentNormalizer and in-memory Qdrant."""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.ingestion.chunkers.tokens import WhitespaceTokenCounter
from app.ingestion.indexer import Indexer
from app.ingestion.normalize import DocumentNormalizer
from app.retrieval.embedder import FakeEmbedder
from app.retrieval.store import QdrantStore


class TestIndexer:
    """Verifies end-to-end normalization, chunking, and indexing pipeline."""

    def test_ingest_structured_catalog(self, test_settings: Settings) -> None:
        normalizer = DocumentNormalizer()
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)
        counter = WhitespaceTokenCounter()

        indexer = Indexer(
            normalizer=normalizer,
            store=store,
            counter=counter,
            chunker="structured",
            target_tokens=250,
            max_tokens=480,
        )

        data_dir = Path("data/raw")
        report = indexer.ingest_all(data_dir=data_dir)

        assert report.total_files > 0
        assert report.total_chunks > 0
        assert "card" in report.chunks_by_type
        assert "spec" in report.chunks_by_type
        # Verify 480 token ceiling across all indexed chunks
        assert report.max_chunk_tokens <= 480

        # Verify Qdrant point count matches report
        qdrant_count = store.count(chunker="structured")
        assert qdrant_count == report.total_chunks

    def test_if_empty_skips_when_populated(self, test_settings: Settings) -> None:
        normalizer = DocumentNormalizer()
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)
        counter = WhitespaceTokenCounter()

        indexer = Indexer(
            normalizer=normalizer,
            store=store,
            counter=counter,
            chunker="structured",
        )

        data_dir = Path("data/raw")
        # First ingest
        r1 = indexer.ingest_all(data_dir=data_dir)
        assert not r1.skipped

        # Second ingest with if_empty=True
        r2 = indexer.ingest_all(data_dir=data_dir, if_empty=True)
        assert r2.skipped
        assert r2.total_chunks == r1.total_chunks

    def test_rebuild_resets_collection(self, test_settings: Settings) -> None:
        normalizer = DocumentNormalizer()
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)
        counter = WhitespaceTokenCounter()

        indexer = Indexer(
            normalizer=normalizer,
            store=store,
            counter=counter,
            chunker="structured",
        )

        data_dir = Path("data/raw")
        r1 = indexer.ingest_all(data_dir=data_dir)
        count_before = store.count("structured")

        r2 = indexer.ingest_all(data_dir=data_dir, rebuild=True)
        count_after = store.count("structured")

        assert count_after == count_before
        assert r2.total_chunks == r1.total_chunks

    def test_ingest_fixed_chunker(self, test_settings: Settings) -> None:
        normalizer = DocumentNormalizer()
        embedder = FakeEmbedder(dim=384)
        store = QdrantStore(settings=test_settings, embedder=embedder)
        counter = WhitespaceTokenCounter()

        indexer = Indexer(
            normalizer=normalizer,
            store=store,
            counter=counter,
            chunker="fixed",
        )

        data_dir = Path("data/raw")
        report = indexer.ingest_all(data_dir=data_dir)

        assert report.total_files > 0
        assert report.total_chunks > 0
        assert "fixed" in report.chunks_by_type
        assert store.count("fixed") == report.total_chunks
