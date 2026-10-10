"""Cross-encoder reranking models for precision re-ordering of retrieval candidates.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from app.core.schemas import Chunk

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Protocol for scoring and reranking chunks against a query."""

    def rerank(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Rerank chunks relative to query and return top_k (Chunk, score) pairs."""
        ...


class CrossEncoderReranker:
    """Production reranker wrapping FastEmbed TextCrossEncoder model."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-12-v2",
        cache_dir: str | None = None,
    ) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            try:
                from fastembed import TextCrossEncoder  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "FastEmbed TextCrossEncoder is not available in the current environment."
                ) from exc

        kwargs: dict[str, Any] = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self._model = TextCrossEncoder(**kwargs)
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        # FastEmbed cross encoder accepts documents as list of strings
        doc_texts = [c.text for c in chunks]
        try:
            scores_generator = self._model.rerank(query, doc_texts)
            results: list[tuple[Chunk, float]] = []
            for i, item in enumerate(scores_generator):
                if isinstance(item, dict):
                    idx = int(item["result"])
                    score = float(item["score"])
                elif hasattr(item, "index") and hasattr(item, "score"):
                    idx = int(item.index)
                    score = float(item.score)
                elif isinstance(item, (int, float)) or hasattr(item, "__float__"):
                    idx = i
                    score = float(item)
                else:
                    try:
                        idx, score = int(item[0]), float(item[1])
                    except Exception:
                        idx, score = i, float(item)
                results.append((chunks[idx], score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        except Exception as exc:
            logger.warning("FastEmbed reranker failed; falling back to original order: %s", exc)
            return [(c, 1.0 / (i + 1)) for i, c in enumerate(chunks[:top_k])]


class FakeReranker:
    """Deterministic offline reranker scoring by token overlap for tests and CI."""

    def rerank(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        import re

        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[Chunk, float]] = []

        for idx, chunk in enumerate(chunks):
            c_tokens = set(re.findall(r"\w+", chunk.text.lower()))
            overlap = len(q_tokens.intersection(c_tokens))
            # Base score derived from overlap plus order preservation penalty
            score = float(overlap * 1.0 + (len(chunks) - idx) * 0.01)
            scored.append((chunk, score))

        scored.sort(key=lambda item: (item[1], item[0].chunk_id), reverse=True)
        return scored[:top_k]
