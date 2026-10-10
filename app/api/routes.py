"""HTTP REST and SSE endpoints for Filumart RAG Assistant.

Exposes:
- GET /health: Vector DB connectivity, LLM provider readiness, health status.
- GET /filters: Distinct categories, suppliers, countries, and product IDs.
- POST /ask: Synchronous question answering (buffered JSON response).
- POST /ask/stream: Real-time Server-Sent Events (SSE) streaming answer.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.api.sse import sse_generator
from app.core.schemas import AskRequest

logger = logging.getLogger("app.api.routes")
router = APIRouter()


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Return system health, vector DB point count, and active LLM provider status."""
    settings = getattr(request.app.state, "settings", None)
    store = getattr(request.app.state, "store", None)
    router_instance = getattr(request.app.state, "router", None)

    # 1. Vector DB health check
    vdb_mode = getattr(settings, "QDRANT_MODE", "unknown") if settings else "unknown"
    collection_count = 0
    reachable = False
    if store is not None:
        try:
            collection_count = store.count()
            reachable = True
        except Exception as exc:
            logger.warning("[health] Vector DB check failed: %s", exc)
            reachable = False

    # 2. LLM readiness check
    llm_mode = getattr(settings, "LLM_MODE", "live") if settings else "live"
    active_providers: list[str] = []
    if router_instance is not None and hasattr(router_instance, "providers"):
        active_providers = [p.name for p in router_instance.providers]
    elif settings is not None:
        if settings.GEMINI_API_KEY:
            active_providers.append("gemini")
        if settings.GROQ_API_KEY:
            active_providers.append("groq")
        if settings.OPENROUTER_API_KEY:
            active_providers.append("openrouter")
        if settings.OLLAMA_BASE_URL:
            active_providers.append("ollama")

    # Degraded if retrieval-only or no active providers configured
    is_degraded = (
        llm_mode == "retrieval_only"
        or len(active_providers) == 0
        or not reachable
    )
    status_str = "degraded" if is_degraded else "ok"

    return JSONResponse(
        status_code=200,
        content={
            "status": status_str,
            "vector_db": {
                "mode": vdb_mode,
                "collection_count": collection_count,
                "reachable": reachable,
            },
            "llm": {
                "mode": llm_mode,
                "active_providers": active_providers,
            },
            "version": "1.0.0",
        },
    )


@router.get("/filters")
async def get_filters(request: Request) -> JSONResponse:
    """Return unique categories, suppliers, countries, and product IDs for UI dropdowns."""
    store = getattr(request.app.state, "store", None)
    if store is None:
        return JSONResponse(
            status_code=200,
            content={
                "categories": [],
                "suppliers": [],
                "countries": [],
                "product_ids": [],
            },
        )

    categories = store.scroll_unique_values("category")
    suppliers = store.scroll_unique_values("supplier_name")
    countries = store.scroll_unique_values("country")
    product_ids = store.scroll_unique_values("product_id")

    return JSONResponse(
        status_code=200,
        content={
            "categories": categories,
            "suppliers": suppliers,
            "countries": countries,
            "product_ids": product_ids,
        },
    )


@router.post("/ask")
async def ask(body: AskRequest, request: Request) -> JSONResponse:
    """Synchronous ask endpoint: buffers complete generation and returns JSON."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "GenerationService is not initialized.",
                }
            },
        )

    request_id = f"req_{uuid.uuid4().hex[:8]}"
    tokens: list[str] = []
    meta: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = []
    grounding: dict[str, Any] | None = None
    done: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    async for event in service.generate(body, request_id=request_id):
        if event.type == "meta":
            meta = event.payload
        elif event.type == "debug":
            debug = event.payload
        elif event.type == "token":
            tokens.append(event.payload.get("content", ""))
        elif event.type == "sources":
            sources = event.payload.get("sources", [])
        elif event.type == "grounding":
            grounding = event.payload
        elif event.type == "done":
            done = event.payload
        elif event.type == "error":
            error = event.payload

    answer = "".join(tokens)
    response_content: dict[str, Any] = {
        "request_id": request_id,
        "answer": answer,
        "sources": sources,
        "grounding": grounding,
        "meta": meta,
        "done": done,
    }
    if debug is not None:
        response_content["debug"] = debug
    if error is not None:
        response_content["error"] = error

    return JSONResponse(status_code=200, content=response_content)


@router.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> EventSourceResponse:
    """Streaming ask endpoint: yields Server-Sent Events per Section 5.4."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="GenerationService is not initialized.",
        )

    settings = getattr(request.app.state, "settings", None)
    heartbeat_s = float(settings.HEARTBEAT_S) if settings else 15.0
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    generator = sse_generator(
        request=request,
        service=service,
        ask_request=body,
        request_id=request_id,
        heartbeat_s=heartbeat_s,
    )
    return EventSourceResponse(generator, ping=int(heartbeat_s))
