"""Embedder implementations: FastEmbed ONNX wrapper and deterministic FakeEmbedder."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from typing import Any, Protocol, runtime_checkable

from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding textual content into dense vector representations."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute dense embeddings for a batch of strings."""
        ...

    @property
    def dim(self) -> int:
        """Dimensionality of vector representations produced by this embedder."""
        ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Protocol for embedding textual content into sparse representations."""

    def embed(self, texts: list[str]) -> list[Any]:
        """Compute sparse embeddings for a batch of strings."""
        ...

    def query_embed(self, query: str) -> Any:
        """Compute sparse embedding for a search query string."""
        ...


class FastEmbedEmbedder:
    """Production embedder wrapping FastEmbed's ONNX runtime model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ) -> None:
        from fastembed import TextEmbedding

        kwargs: dict[str, Any] = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self._model = TextEmbedding(**kwargs)
        self._model_name = model_name
        self._dim = 384  # Standard bge-small-en-v1.5 embedding dimensionality

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = list(self._model.embed(texts))
        return [
            e.tolist() if hasattr(e, "tolist") else list(e)
            for e in embeddings
        ]

    @property
    def dim(self) -> int:
        return self._dim


class FakeEmbedder:
    """Deterministic hash-based pseudo-embedder for offline testing and CI runs."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec: list[float] = []
            for i in range(self._dim):
                byte_val = h[i % len(h)]
                # Mix byte value with index to prevent repeating dimensions
                val = (((byte_val ^ (i & 0xFF)) / 255.0) * 2.0) - 1.0
                vec.append(val)
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            results.append(vec)
        return results

    @property
    def dim(self) -> int:
        return self._dim


class FastEmbedSparseEmbedder:
    """Production sparse embedder wrapping FastEmbed's BM25 model."""

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        cache_dir: str | None = None,
    ) -> None:
        try:
            from fastembed import SparseTextEmbedding
        except ImportError:
            from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding  # type: ignore

        kwargs: dict[str, Any] = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self._model = SparseTextEmbedding(**kwargs)
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[Any]:
        if not texts:
            return []
        results = list(self._model.embed(texts))
        sparse_vecs: list[SparseVector] = []
        for item in results:
            indices = (
                item.indices.tolist()
                if hasattr(item.indices, "tolist")
                else list(item.indices)
            )
            values = item.values.tolist() if hasattr(item.values, "tolist") else list(item.values)
            sparse_vecs.append(SparseVector(indices=indices, values=values))
        return sparse_vecs

    def query_embed(self, query: str) -> Any:
        if hasattr(self._model, "query_embed"):
            results = list(self._model.query_embed([query]))
        else:
            results = list(self._model.embed([query]))
        if not results:
            return SparseVector(indices=[], values=[])
        item = results[0]
        indices = (
            item.indices.tolist()
            if hasattr(item.indices, "tolist")
            else list(item.indices)
        )
        values = item.values.tolist() if hasattr(item.values, "tolist") else list(item.values)
        return SparseVector(indices=indices, values=values)


class FakeSparseEmbedder:
    """Deterministic hash-based sparse pseudo-embedder for offline testing."""

    def _tokenize_to_sparse(self, text: str) -> Any:
        tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        if not tokens:
            return SparseVector(indices=[], values=[])
        counts = Counter(tokens)
        token_indices: dict[int, float] = {}
        for tok, count in counts.items():
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
            idx = (h % 16777215) + 1
            weight = float(1.0 + (count - 1) * 0.5)
            token_indices[idx] = weight
        sorted_indices = sorted(token_indices.keys())
        values = [token_indices[idx] for idx in sorted_indices]
        return SparseVector(indices=sorted_indices, values=values)

    def embed(self, texts: list[str]) -> list[Any]:
        return [self._tokenize_to_sparse(t) for t in texts]

    def query_embed(self, query: str) -> Any:
        return self._tokenize_to_sparse(query)

