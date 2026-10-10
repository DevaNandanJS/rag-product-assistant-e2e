"""Token counter using FastEmbed's tokenizer, with whitespace fallback."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""
        ...


class FastEmbedTokenCounter:
    """Counts tokens using the tokenizer from the FastEmbed model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._tokenizer: Any = None

    def _load(self) -> None:
        from fastembed import TextEmbedding

        kwargs: dict[str, Any] = {"model_name": self._model_name}
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir
        model = TextEmbedding(**kwargs)

        if hasattr(model, "tokenizer"):
            self._tokenizer = model.tokenizer
        elif hasattr(model, "model") and hasattr(model.model, "tokenizer"):
            self._tokenizer = model.model.tokenizer
        elif hasattr(model, "_model") and hasattr(model._model, "tokenizer"):
            self._tokenizer = model._model.tokenizer
        else:
            raise AttributeError("Could not resolve tokenizer attribute in FastEmbed TextEmbedding")

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        if self._tokenizer is None:
            self._load()

        if hasattr(self._tokenizer, "encode"):
            encoded = self._tokenizer.encode(text)
            if hasattr(encoded, "ids"):
                return len(encoded.ids)
            if isinstance(encoded, list):
                return len(encoded)

        if callable(self._tokenizer):
            encoded = self._tokenizer(text)
            if isinstance(encoded, dict) and "input_ids" in encoded:
                return len(encoded["input_ids"])

        words = text.split()
        return max(1, int(len(words) / 0.75))


class WhitespaceTokenCounter:
    """Fallback: approximate token count via whitespace splitting (÷ 0.75)."""

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        words = text.split()
        return max(1, int(len(words) / 0.75))


def get_token_counter(
    model_name: str = "BAAI/bge-small-en-v1.5",
    cache_dir: str | None = None,
) -> TokenCounter:
    """Factory: returns FastEmbed counter, falls back to whitespace if unavailable."""
    try:
        counter = FastEmbedTokenCounter(model_name=model_name, cache_dir=cache_dir)
        counter.count("test tokenization")
        return counter
    except Exception as exc:
        logger.warning("FastEmbed tokenizer unavailable (%s); using whitespace fallback.", exc)
        return WhitespaceTokenCounter()
