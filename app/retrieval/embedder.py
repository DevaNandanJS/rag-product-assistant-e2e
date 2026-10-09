"""Embedder implementations: FastEmbed ONNX wrapper and deterministic FakeEmbedder."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, runtime_checkable

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
