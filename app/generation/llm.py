"""LLM provider protocol and OpenAI-compatible streaming client.

Wraps the async openai.AsyncOpenAI client with:
- Two-phase timeout: first-token deadline and per-chunk idle deadline.
- Guaranteed upstream generator closure in finally blocks.
- Re-raises openai.RateLimitError as-is (for router cooldown logic).
- Re-raises asyncio.TimeoutError as LLMTimeoutError.

No new dependencies — uses only stdlib asyncio and the pinned openai package.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for streaming text generation providers."""

    @property
    def name(self) -> str:
        """Human-readable provider identifier (e.g. 'gemini', 'groq')."""
        ...

    @property
    def model(self) -> str:
        """Model identifier string sent in the API request."""
        ...

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream response tokens for the given message list.

        Yields individual token strings as they arrive from the upstream
        provider socket. Must never buffer the complete response.

        Raises:
            LLMTimeoutError: First token not received within first_token_timeout_s,
                or no token received within idle_timeout_s between chunks.
            openai.RateLimitError: Provider responded with HTTP 429.
            LLMUnavailableError: Provider is unreachable or returned a non-retryable error.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# OpenAI-compatible provider implementation
# ---------------------------------------------------------------------------


class OpenAICompatProvider:
    """Streams tokens from any OpenAI-compatible endpoint using the async openai client.

    Compatible with: Google Gemini (v1beta/openai/), Groq, OpenRouter, Ollama.
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        first_token_timeout_s: float = 20.0,
        idle_timeout_s: float = 20.0,
    ) -> None:
        self._name = name
        self._model = model
        self._first_token_timeout_s = first_token_timeout_s
        self._idle_timeout_s = idle_timeout_s

        # Lazy import: openai is already in requirements.txt
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required. Install it via: pip install openai"
            ) from exc

        self._client: Any = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream response tokens, enforcing two-phase timeouts.

        Phase 1 — First-Token Timeout:
            asyncio.wait_for() wraps the first __anext__() call on the openai
            streaming iterator. If the server does not send a token within
            `first_token_timeout_s`, asyncio.TimeoutError is raised and
            re-raised as LLMTimeoutError.

        Phase 2 — Idle Timeout:
            Each subsequent __anext__() call is wrapped in wait_for() with
            `idle_timeout_s`. The deadline resets on every received chunk,
            so a slow-but-alive stream is never prematurely terminated.

        The openai stream is always closed in a `finally` block to prevent
        resource leaks regardless of how the caller exits (cancellation,
        exception, normal return).
        """
        from app.core.errors import LLMTimeoutError, LLMUnavailableError

        openai_stream: Any = None
        try:
            openai_stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            aiter = openai_stream.__aiter__()

            # --- Phase 1: first token ---
            try:
                chunk = await asyncio.wait_for(
                    aiter.__anext__(),
                    timeout=self._first_token_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                raise LLMTimeoutError(
                    f"[{self._name}] First token not received within "
                    f"{self._first_token_timeout_s:.0f}s."
                ) from exc
            except StopAsyncIteration:
                return

            # Yield first token if it has content
            content = _extract_content(chunk)
            if content:
                yield content

            # --- Phase 2: idle timeout per subsequent chunk ---
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        aiter.__anext__(),
                        timeout=self._idle_timeout_s,
                    )
                except asyncio.TimeoutError as exc:
                    raise LLMTimeoutError(
                        f"[{self._name}] No token received for "
                        f"{self._idle_timeout_s:.0f}s (idle timeout)."
                    ) from exc
                except StopAsyncIteration:
                    break

                content = _extract_content(chunk)
                if content:
                    yield content

        except (LLMTimeoutError,):
            raise  # Propagate for router failover decision
        except Exception as exc:
            # Preserve openai.RateLimitError for router cooldown logic
            _class_name = type(exc).__name__
            if "RateLimitError" in _class_name:
                raise
            # Anything else becomes LLMUnavailableError
            logger.warning("[%s] Provider error: %s: %s", self._name, _class_name, exc)
            raise LLMUnavailableError(
                f"[{self._name}] Provider error: {_class_name}: {exc}"
            ) from exc
        finally:
            if openai_stream is not None:
                try:
                    await openai_stream.close()
                except Exception:  # noqa: BLE001
                    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_content(chunk: Any) -> str:
    """Extract token string from an openai streaming ChatCompletionChunk."""
    try:
        delta = chunk.choices[0].delta
        return delta.content or ""
    except (AttributeError, IndexError):
        return ""
