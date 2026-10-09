"""Qdrant vector store management: collection provisioning, payload indexing,
and idempotent document upsert/replacement.
"""

from __future__ import annotations

import logging
from typing import Sequence

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
from app.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


class QdrantStore:
    """Manages Qdrant vector storage, collection creation, and document updates."""

    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self.settings = settings
        self.embedder = embedder
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
            raise VectorDBError(f"Failed to initialize QdrantClient in mode '{mode}': {exc}") from exc

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
                warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")
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

            # Build PointStruct items
            points: list[PointStruct] = []
            for chunk, vec in zip(chunks, vectors):
                points.append(
                    PointStruct(
                        id=chunk.point_id,
                        vector={"dense": vec},
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
            raise VectorDBError(f"Failed to count points in collection '{col_name}': {exc}") from exc

    def delete_collection(self, chunker: str | None = None) -> None:
        """Delete collection if it exists."""
        col_name = self.collection_name(chunker)
        if self.collection_exists(col_name):
            try:
                self.client.delete_collection(collection_name=col_name)
            except Exception as exc:
                raise VectorDBError(f"Failed deleting collection '{col_name}': {exc}") from exc
