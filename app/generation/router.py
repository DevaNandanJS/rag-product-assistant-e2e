"""LLM provider router with priority failover, 429 cooldown, and retrieval_only mode.

Priority order: gemini -> groq -> openrouter -> ollama
(determined by the LLM_PROVIDERS config and which providers are configured)

Failover rules (Section 8 / ADR-08):
- Failover is ONLY permitted BEFORE the first token is emitted to the caller.
- Once the first token has been yielded, errors propagate immediately.
- On HTTP 429 (RateLimitError): provider placed on 60-second cooldown.
- On LLMTimeoutError: skip to next provider (pre-first-token only).
- If all providers are exhausted: raise LLMUnavailableError.
- If mode == 'retrieval_only': yield nothing immediately (no providers tried).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

_COOLDOWN_SECONDS = 60.0


class LLMRouter:
    """Priority-ordered provider router with automatic failover and cooldown management."""

    def __init__(
        self,
        providers: list[Any],  # list[LLMProvider] — typed loosely to avoid Protocol issues
        mode: str = "live",
    ) -> None:
        """Initialize the router.

        Args:
            providers: Ordered list of LLMProvider instances. Providers with no
                configured API key should be excluded before passing here.
            mode: 'live' (attempt providers), 'retrieval_only' (yield nothing),
                  or 'replay' (handled upstream by GenerationService cache).
        """
        self._providers = providers
        self._mode = mode.lower()
        # Maps provider name -> monotonic timestamp when cooldown expires
        self._cooldowns: dict[str, float] = {}

    @property
    def active_provider_names(self) -> list[str]:
        """Names of all providers registered with this router."""
        return [p.name for p in self._providers]

    def _is_on_cooldown(self, provider_name: str) -> bool:
        expiry = self._cooldowns.get(provider_name, 0.0)
        return time.monotonic() < expiry

    def _set_cooldown(self, provider_name: str) -> None:
        expiry = time.monotonic() + _COOLDOWN_SECONDS
        self._cooldowns[provider_name] = expiry
        logger.warning(
            "[router] Provider '%s' placed on %ds cooldown (429 rate limit).",
            provider_name,
            _COOLDOWN_SECONDS,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream tokens from the highest-priority available provider.

        Implements the failover state machine per ADR-08 and Section 8 Phase 6.
        """
        from app.core.errors import LLMTimeoutError, LLMUnavailableError

        if self._mode == "retrieval_only":
            logger.info("[router] Mode is 'retrieval_only' — skipping all LLM providers.")
            return

        if not self._providers:
            raise LLMUnavailableError(
                "No LLM providers are configured. "
                "Set GEMINI_API_KEY, GROQ_API_KEY, or another provider key in .env, "
                "or set LLM_MODE=retrieval_only."
            )

        last_error: Exception | None = None

        for provider in self._providers:
            pname = provider.name

            if self._is_on_cooldown(pname):
                remaining = self._cooldowns[pname] - time.monotonic()
                logger.info(
                    "[router] Skipping '%s' — on cooldown for %.0fs.", pname, remaining
                )
                continue

            logger.info("[router] Attempting provider '%s' (model: %s).", pname, provider.model)
            first_token_emitted = False

            try:
                async for token in provider.stream(messages, max_tokens, temperature):
                    if not first_token_emitted:
                        first_token_emitted = True
                    yield token

                # Stream completed successfully
                return

            except Exception as exc:
                _class_name = type(exc).__name__

                # --- Rate limit: cooldown and failover ---
                if "RateLimitError" in _class_name:
                    if first_token_emitted:
                        # Stream already started — cannot failover cleanly
                        logger.error(
                            "[router] '%s' hit 429 after first token — cannot failover. "
                            "Propagating error.",
                            pname,
                        )
                        self._set_cooldown(pname)
                        raise LLMUnavailableError(
                            f"Provider '{pname}' hit rate limit mid-stream."
                        ) from exc
                    self._set_cooldown(pname)
                    last_error = exc
                    continue

                # --- Timeout before first token: failover ---
                if isinstance(exc, LLMTimeoutError):
                    if first_token_emitted:
                        logger.error(
                            "[router] '%s' idle timeout after first token — propagating.",
                            pname,
                        )
                        raise
                    logger.warning(
                        "[router] '%s' timed out before first token — trying next provider.",
                        pname,
                    )
                    last_error = exc
                    continue

                # --- Any other error before first token: failover ---
                if not first_token_emitted:
                    logger.warning(
                        "[router] '%s' raised %s before first token — trying next provider: %s",
                        pname,
                        _class_name,
                        exc,
                    )
                    last_error = exc
                    continue

                # Error after first token: propagate immediately
                raise

        # All providers exhausted
        detail = f" Last error: {last_error}" if last_error else ""
        raise LLMUnavailableError(
            f"All configured LLM providers are unavailable or on cooldown.{detail} "
            "Set LLM_MODE=retrieval_only to run without generation."
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_router_from_settings(settings: Any) -> LLMRouter:
    """Construct an LLMRouter populated with providers derived from Settings.

    Providers with empty API keys are skipped unless they are Ollama
    (which operates without a key on a local endpoint).
    """
    from app.generation.llm import OpenAICompatProvider

    priority = settings.get_providers_list()
    providers: list[Any] = []

    for pname in priority:
        if pname == "gemini":
            if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY not in ("", "your_key_here"):
                providers.append(
                    OpenAICompatProvider(
                        name="gemini",
                        api_key=settings.GEMINI_API_KEY,
                        base_url=settings.GEMINI_BASE_URL,
                        model=settings.GEMINI_MODEL,
                        first_token_timeout_s=settings.LLM_FIRST_TOKEN_TIMEOUT_S,
                        idle_timeout_s=settings.LLM_IDLE_TIMEOUT_S,
                    )
                )
            else:
                logger.info("[router] Skipping 'gemini' — GEMINI_API_KEY not configured.")

        elif pname == "groq":
            if settings.GROQ_API_KEY and settings.GROQ_API_KEY not in ("", "your_key_here"):
                providers.append(
                    OpenAICompatProvider(
                        name="groq",
                        api_key=settings.GROQ_API_KEY,
                        base_url=settings.GROQ_BASE_URL,
                        model=settings.GROQ_MODEL,
                        first_token_timeout_s=settings.LLM_FIRST_TOKEN_TIMEOUT_S,
                        idle_timeout_s=settings.LLM_IDLE_TIMEOUT_S,
                    )
                )
            else:
                logger.info("[router] Skipping 'groq' — GROQ_API_KEY not configured.")

        elif pname == "openrouter":
            if settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_KEY not in (
                "",
                "your_key_here",
            ):
                providers.append(
                    OpenAICompatProvider(
                        name="openrouter",
                        api_key=settings.OPENROUTER_API_KEY,
                        base_url=settings.OPENROUTER_BASE_URL,
                        model=settings.OPENROUTER_MODEL,
                        first_token_timeout_s=settings.LLM_FIRST_TOKEN_TIMEOUT_S,
                        idle_timeout_s=settings.LLM_IDLE_TIMEOUT_S,
                    )
                )
            else:
                logger.info("[router] Skipping 'openrouter' — OPENROUTER_API_KEY not configured.")

        elif pname == "ollama":
            if settings.OLLAMA_BASE_URL:
                providers.append(
                    OpenAICompatProvider(
                        name="ollama",
                        api_key="ollama",  # Ollama ignores auth
                        base_url=settings.OLLAMA_BASE_URL,
                        model=settings.OLLAMA_MODEL,
                        first_token_timeout_s=settings.LLM_FIRST_TOKEN_TIMEOUT_S,
                        idle_timeout_s=settings.LLM_IDLE_TIMEOUT_S,
                    )
                )
            else:
                logger.info("[router] Skipping 'ollama' — OLLAMA_BASE_URL not configured.")

    mode = settings.LLM_MODE
    if not providers and mode == "live":
        logger.warning(
            "[router] No LLM providers configured in 'live' mode. "
            "Effective mode will be 'retrieval_only'."
        )
        mode = "retrieval_only"

    return LLMRouter(providers=providers, mode=mode)
