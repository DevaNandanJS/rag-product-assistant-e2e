"""GenerationService orchestrator coordinating retrieval, gating, prompting, LLM streaming, and post-check.

Emits structured SSE GenerationEvent objects adhering to Section 5.4 of BUILD_GUIDE.md.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.errors import LLMUnavailableError
from app.core.schemas import AskRequest, SourceItem
from app.generation.cache import GenerationCache
from app.generation.postcheck import verify
from app.generation.prompts import build_messages
from app.generation.router import LLMRouter
from app.retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationEvent:
    """Standardized event frame yielded by GenerationService."""

    type: str  # meta | debug | token | sources | grounding | error | done
    payload: dict[str, Any]


class GenerationService:
    """Orchestrates end-to-end question answering pipeline."""

    def __init__(
        self,
        settings: Settings,
        pipeline: RetrievalPipeline,
        router: LLMRouter,
        cache: GenerationCache | None = None,
    ) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._router = router
        self._cache = cache

    async def generate(
        self,
        request: AskRequest,
        request_id: str | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        """Execute grounded generation and yield SSE event sequence.

        Event sequence (Section 5.4):
        1. meta
        2. debug (if request.debug)
        3. token (repeated for each token / chunk)
        4. sources
        5. grounding
        6. done
        """
        t_start = time.perf_counter()
        req_id = request_id or f"req_{uuid.uuid4().hex[:8]}"

        # Identify primary active provider for metadata
        providers = getattr(self._router, "_providers", [])
        if providers:
            p = providers[0]
            p_name = getattr(p, "name", "unknown")
            p_model = getattr(p, "model", "unknown")
            provider_label = f"{p_name}:{p_model}"
            provider_model = p_model
        else:
            provider_label = "none"
            provider_model = "none"

        try:
            # 1. Retrieval
            t_retrieval_start = time.perf_counter()
            qdrant_filter = None
            if request.filters:
                qdrant_filter = {
                    k: v
                    for k, v in request.filters.model_dump().items()
                    if v is not None
                }

            retrieval_result = self._pipeline.retrieve(
                question=request.question,
                filters=qdrant_filter,
                top_k=request.top_k or self._settings.TOP_K,
                threshold=request.threshold,
            )
            retrieval_latency_ms = (time.perf_counter() - t_retrieval_start) * 1000

            # 2. Yield 'meta' event
            yield GenerationEvent(
                type="meta",
                payload={
                    "type": "meta",
                    "request_id": req_id,
                    "mode": self._settings.LLM_MODE,
                    "provider": provider_label,
                    "retrieval": {
                        "mode": self._settings.RETRIEVAL_MODE,
                        "k": len(retrieval_result.chunks),
                        "entities": retrieval_result.entities,
                        "filters": (
                            request.filters.model_dump(exclude_none=True)
                            if request.filters
                            else {}
                        ),
                        "gate": {
                            "passed": retrieval_result.gate.passed,
                            "signal": retrieval_result.gate.signal,
                            "value": (
                                round(retrieval_result.gate.score, 4)
                                if retrieval_result.gate.score is not None
                                else None
                            ),
                            "threshold": retrieval_result.gate.threshold,
                        },
                        "latency_ms": round(retrieval_latency_ms, 2),
                    },
                },
            )

            # 3. Yield optional 'debug' event
            if request.debug:
                yield GenerationEvent(
                    type="debug",
                    payload={
                        "type": "debug",
                        "dense_ranks": retrieval_result.debug_info.get("dense_ranks", []),
                        "sparse_ranks": retrieval_result.debug_info.get("sparse_ranks", []),
                        "fused_ranks": retrieval_result.debug_info.get("fused_ranks", []),
                    },
                )

            # 4. Check gate rejection
            if not retrieval_result.gate.passed:
                refusal_msg = "Not documented in the provided knowledge base."
                yield GenerationEvent(
                    type="token",
                    payload={"type": "token", "content": refusal_msg},
                )
                yield GenerationEvent(
                    type="sources",
                    payload={"type": "sources", "sources": []},
                )
                yield GenerationEvent(
                    type="grounding",
                    payload={"type": "grounding", "ok": True, "warnings": []},
                )
                total_latency_ms = (time.perf_counter() - t_start) * 1000
                yield GenerationEvent(
                    type="done",
                    payload={
                        "type": "done",
                        "status": "gated",
                        "latency_ms": round(total_latency_ms, 2),
                        "ttft_ms": round(retrieval_latency_ms, 2),
                    },
                )
                return

            # 5. Build prompt messages
            messages = build_messages(
                question=request.question,
                chunks=retrieval_result.chunks,
                is_comparison=retrieval_result.is_comparison,
                history=request.history,
            )

            accumulated_tokens: list[str] = []
            ttft_ms: float | None = None

            # 6. Check retrieval-only mode
            if self._settings.LLM_MODE == "retrieval_only":
                canned_fallback = (
                    "[Retrieval-only mode] Top passages retrieved above. "
                    "Add an LLM API key to enable answer generation."
                )
                ttft_ms = (time.perf_counter() - t_start) * 1000
                accumulated_tokens.append(canned_fallback)
                yield GenerationEvent(
                    type="token",
                    payload={"type": "token", "content": canned_fallback},
                )
            else:
                # Check cache
                cache_key = GenerationCache.make_key(messages, provider_model)
                cached_answer = (
                    self._cache.lookup(cache_key) if self._cache is not None else None
                )

                if cached_answer is not None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                    accumulated_tokens.append(cached_answer)
                    yield GenerationEvent(
                        type="token",
                        payload={"type": "token", "content": cached_answer},
                    )
                else:
                    try:
                        async for token in self._router.stream(
                            messages=messages,
                            max_tokens=self._settings.LLM_MAX_TOKENS,
                            temperature=self._settings.LLM_TEMPERATURE,
                        ):
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t_start) * 1000
                            accumulated_tokens.append(token)
                            yield GenerationEvent(
                                type="token",
                                payload={"type": "token", "content": token},
                            )
                    except LLMUnavailableError as exc:
                        logger.warning(
                            "[generation] LLM unavailable: %s. Emitting retrieval-only fallback.",
                            exc,
                        )
                        canned_fallback = (
                            "[Retrieval-only mode] Top passages retrieved above. "
                            "Add an LLM API key to enable answer generation."
                        )
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        accumulated_tokens.append(canned_fallback)
                        yield GenerationEvent(
                            type="token",
                            payload={"type": "token", "content": canned_fallback},
                        )

            full_text = "".join(accumulated_tokens)

            # 7. Build sources event
            source_items: list[dict[str, Any]] = []
            source_ids: set[str] = set()
            for i, chunk in enumerate(retrieval_result.chunks, 1):
                sid = f"S{i}"
                source_ids.add(sid)
                is_cited = bool(re.search(rf"\[{sid}\]", full_text))
                raw_text = chunk.display_text if chunk.display_text else chunk.text
                snippet = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text

                item = SourceItem(
                    id=sid,
                    product_id=chunk.product_id,
                    product_name=chunk.product_name,
                    document=chunk.document,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    source_type=chunk.source_type,
                    ocr_confidence=chunk.ocr_confidence,
                    cited=is_cited,
                    suspicious=chunk.suspicious,
                    snippet=snippet,
                )
                source_items.append(item.model_dump())

            yield GenerationEvent(
                type="sources",
                payload={"type": "sources", "sources": source_items},
            )

            # 8. Run deterministic claim verification
            context_str = " ".join(
                (c.display_text or c.text) for c in retrieval_result.chunks
            )
            warnings = verify(
                generated=full_text,
                context=context_str,
                source_ids=source_ids,
            )
            yield GenerationEvent(
                type="grounding",
                payload={
                    "type": "grounding",
                    "ok": len(warnings) == 0,
                    "warnings": [w.claim for w in warnings],
                },
            )

            # 9. Optionally write to cache
            if (
                self._cache is not None
                and self._settings.LLM_CACHE_WRITE
                and full_text
                and self._settings.LLM_MODE != "retrieval_only"
            ):
                self._cache.write(cache_key, full_text)

            # 10. Yield done event
            total_latency_ms = (time.perf_counter() - t_start) * 1000
            yield GenerationEvent(
                type="done",
                payload={
                    "type": "done",
                    "status": "ok",
                    "latency_ms": round(total_latency_ms, 2),
                    "ttft_ms": round(
                        ttft_ms if ttft_ms is not None else total_latency_ms, 2
                    ),
                },
            )

        except Exception as exc:
            logger.exception("[generation] Unhandled error during generation: %s", exc)
            yield GenerationEvent(
                type="error",
                payload={
                    "type": "error",
                    "code": "GENERATION_ERROR",
                    "message": str(exc),
                    "retryable": False,
                },
            )
            total_latency_ms = (time.perf_counter() - t_start) * 1000
            yield GenerationEvent(
                type="done",
                payload={
                    "type": "done",
                    "status": "error",
                    "latency_ms": round(total_latency_ms, 2),
                    "ttft_ms": 0.0,
                },
            )
