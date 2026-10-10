"""Qdrant vector store management: collection provisioning, payload indexing,
and idempotent document upsert/replacement.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import Settings
from app.core.errors import VectorDBError
from app.core.schemas import Chunk
from app.retrieval.embedder import (
    Embedder,
    FakeEmbedder,
    FakeSparseEmbedder,
    FastEmbedSparseEmbedder,
    SparseEmbedder,
)

logger = logging.getLogger(__name__)


class QdrantStore:
    """Manages Qdrant vector storage, collection creation, and document updates."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        if sparse_embedder is not None:
            self.sparse_embedder = sparse_embedder
        elif isinstance(embedder, FakeEmbedder):
            self.sparse_embedder = FakeSparseEmbedder()
        else:
            self.sparse_embedder = FastEmbedSparseEmbedder(
                cache_dir=settings.FASTEMBED_CACHE_PATH
            )
        self.client = self._init_client()

    def _init_client(self) -> QdrantClient:
        """Initialize Qdrant client based on settings.QDRANT_MODE."""
        mode = self.settings.QDRANT_MODE
        try:
            if mode == "memory":
                return QdrantClient(location=":memory:")
            elif mode == "local":
                return QdrantClient(path=self.settings.QDRANT_PATH)
            elif mode == "server":
                return QdrantClient(url=self.settings.QDRANT_URL)
            else:
                raise VectorDBError(f"Unsupported QDRANT_MODE: '{mode}'")
        except Exception as exc:
            if isinstance(exc, VectorDBError):
                raise
            msg = f"Failed to initialize QdrantClient in mode '{mode}': {exc}"
            raise VectorDBError(msg) from exc

    def collection_name(self, chunker: str | None = None) -> str:
        """Derive standard collection name matching configuration and chunker type."""
        active_chunker = chunker or self.settings.CHUNKER
        model_slug = self.settings.EMBEDDING_MODEL.replace("/", "--").replace(" ", "_")
        return f"{self.settings.COLLECTION_PREFIX}__{active_chunker}__{model_slug}"

    def collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection exists in Qdrant."""
        try:
            if hasattr(self.client, "collection_exists"):
                return bool(self.client.collection_exists(collection_name))
            collections = [c.name for c in self.client.get_collections().collections]
            return collection_name in collections
        except Exception as exc:
            raise VectorDBError(f"Error checking collection existence: {exc}") from exc

    def ensure_collection(self, chunker: str | None = None) -> None:
        """Provision collection with named dense vector config and keyword payload indexes."""
        col_name = self.collection_name(chunker)
        if self.collection_exists(col_name):
            return

        try:
            self.client.create_collection(
                collection_name=col_name,
                vectors_config={
                    "dense": VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                },
            )

            import warnings

            payload_fields = (
                "product_id",
                "category",
                "supplier_name",
                "country",
                "document",
                "source_type",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                for field in payload_fields:
                    self.client.create_payload_index(
                        collection_name=col_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
            logger.info("Created Qdrant collection '%s' with payload indexes.", col_name)
        except Exception as exc:
            raise VectorDBError(f"Failed to provision collection '{col_name}': {exc}") from exc

    def replace_document(
        self,
        doc_name: str,
        chunks: Sequence[Chunk],
        chunker: str | None = None,
    ) -> int:
        """Idempotently replace all points for a document in Qdrant.

        1. Ensures collection exists.
        2. Deletes existing points matching ``document == doc_name``.
        3. Computes dense embeddings for new chunks.
        4. Upserts new points in batches.

        Returns:
            Number of points upserted.
        """
        col_name = self.collection_name(chunker)
        self.ensure_collection(chunker)

        try:
            # Delete points previously ingested for this document
            doc_filter = Filter(
                must=[FieldCondition(key="document", match=MatchValue(value=doc_name))]
            )
            self.client.delete(collection_name=col_name, points_selector=doc_filter)

            if not chunks:
                return 0

            # Batch embed chunk text representations
            texts = [c.text for c in chunks]
            vectors = self.embedder.embed(texts)
            sparse_vectors = self.sparse_embedder.embed(texts)

            # Build PointStruct items
            points: list[PointStruct] = []
            for chunk, vec, s_vec in zip(chunks, vectors, sparse_vectors, strict=True):
                points.append(
                    PointStruct(
                        id=chunk.point_id,
                        vector={"dense": vec, "sparse": s_vec},
                        payload=chunk.model_dump(),
                    )
                )

            # Upsert in chunks of 64
            batch_size = 64
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=col_name, points=batch)

            return len(points)
        except Exception as exc:
            if isinstance(exc, VectorDBError):
                raise
            raise VectorDBError(
                f"Failed replacing document '{doc_name}' in collection '{col_name}': {exc}"
            ) from exc

    def count(self, chunker: str | None = None) -> int:
        """Return total number of points in the active collection."""
        col_name = self.collection_name(chunker)
        if not self.collection_exists(col_name):
            return 0
        try:
            return self.client.count(collection_name=col_name).count
        except Exception as exc:
            msg = f"Failed to count points in collection '{col_name}': {exc}"
            raise VectorDBError(msg) from exc

    def delete_collection(self, chunker: str | None = None) -> None:
        """Delete collection if it exists."""
        col_name = self.collection_name(chunker)
        if self.collection_exists(col_name):
            try:
                self.client.delete_collection(collection_name=col_name)
            except Exception as exc:
                raise VectorDBError(f"Failed deleting collection '{col_name}': {exc}") from exc

    def _build_filter(self, filters: Filter | dict[str, Any] | None) -> Filter | None:
        """Construct Qdrant Filter from dict or return existing Filter."""
        if filters is None:
            return None
        if isinstance(filters, Filter):
            return filters
        if not isinstance(filters, dict):
            return None
        conditions = [
            FieldCondition(key=str(k), match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions) if conditions else None

    def dense_search_with_scores(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        chunker: str | None = None,
        filters: Filter | dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Perform dense vector search, returning (Chunk, similarity_score) pairs.

        Args:
            query_vector: Dense embedding vector for the query.
            top_k: Maximum number of points to retrieve.
            chunker: Chunker name (e.g., 'fixed', 'structured') or None for configured default.
            filters: Optional Qdrant Filter or dict of keyword field-matches.

        Returns:
            List of (Chunk, score) tuples in descending similarity order.
        """
        col_name = self.collection_name(chunker)
        if not self.collection_exists(col_name):
            logger.warning("Search called on nonexistent collection '%s'.", col_name)
            return []

        try:
            vec = list(query_vector)
            query_filter = self._build_filter(filters)
            try:
                res = self.client.query_points(
                    collection_name=col_name,
                    query=vec,
                    using="dense",
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )
                scored_points = res.points
            except Exception:
                scored_points = self.client.search(
                    collection_name=col_name,
                    query_vector=("dense", vec),
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )

            results: list[tuple[Chunk, float]] = []
            for point in scored_points:
                if point.payload:
                    chunk = Chunk.model_validate(point.payload)
                    score = float(point.score) if point.score is not None else 0.0
                    results.append((chunk, score))
            return results
        except Exception as exc:
            raise VectorDBError(
                f"Dense search failed in collection '{col_name}': {exc}"
            ) from exc

    def dense_search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        chunker: str | None = None,
        filters: Filter | dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Perform dense vector search returning matching Chunk instances."""
        return [
            chunk
            for chunk, _ in self.dense_search_with_scores(
                query_vector, top_k, chunker, filters=filters
            )
        ]

    def sparse_search_with_scores(
        self,
        query_text: str,
        top_k: int = 5,
        chunker: str | None = None,
        filters: Filter | dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Perform sparse BM25 vector search, returning (Chunk, sparse_score) pairs.

        Args:
            query_text: Query text to tokenize into sparse BM25 vector.
            top_k: Maximum number of points to retrieve.
            chunker: Chunker name or None for configured default.
            filters: Optional Qdrant Filter or dict of keyword field-matches.

        Returns:
            List of (Chunk, score) tuples in descending score order.
        """
        col_name = self.collection_name(chunker)
        if not self.collection_exists(col_name):
            logger.warning("Search called on nonexistent collection '%s'.", col_name)
            return []

        try:
            sparse_vec = self.sparse_embedder.query_embed(query_text)
            query_filter = self._build_filter(filters)
            try:
                res = self.client.query_points(
                    collection_name=col_name,
                    query=sparse_vec,
                    using="sparse",
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )
                scored_points = res.points
            except Exception:
                scored_points = self.client.search(
                    collection_name=col_name,
                    query_vector=("sparse", sparse_vec),
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )

            results: list[tuple[Chunk, float]] = []
            for point in scored_points:
                if point.payload:
                    chunk = Chunk.model_validate(point.payload)
                    score = float(point.score) if point.score is not None else 0.0
                    results.append((chunk, score))
            return results
        except Exception as exc:
            raise VectorDBError(
                f"Sparse search failed in collection '{col_name}': {exc}"
            ) from exc

    def sparse_search(
        self,
        query_text: str,
        top_k: int = 5,
        chunker: str | None = None,
        filters: Filter | dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Perform sparse vector search returning matching Chunk instances."""
        return [
            chunk
            for chunk, _ in self.sparse_search_with_scores(
                query_text, top_k, chunker, filters=filters
            )
        ]

    def scroll_unique_values(
        self, field: str, chunker: str | None = None, limit: int = 500
    ) -> list[str]:
        """Scroll collection to find distinct non-null string values for a payload field."""
        col_name = self.collection_name(chunker)
        if not self.collection_exists(col_name):
            return []
        seen: set[str] = set()
        offset = None
        try:
            while True:
                results, next_offset = self.client.scroll(
                    collection_name=col_name,
                    limit=100,
                    offset=offset,
                    with_payload=[field],
                )
                for pt in results:
                    val = (pt.payload or {}).get(field)
                    if isinstance(val, str) and val.strip():
                        seen.add(val.strip())
                if next_offset is None or len(seen) >= limit:
                    break
                offset = next_offset
            return sorted(seen)
        except Exception as exc:
            logger.warning("Error scrolling unique values for '%s': %s", field, exc)
            return sorted(seen)


