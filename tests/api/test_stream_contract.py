"""Tests for SSE stream contracts, REST endpoints, and input validation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.core.config import Settings
from app.generation.service import GenerationEvent


def parse_sse_events(raw_text: str) -> list[dict[str, Any]]:
    """Parse raw SSE response body into a list of {event: str, payload: dict}."""
    events: list[dict[str, Any]] = []
    blocks = raw_text.replace("\r\n", "\n").split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        event_name = None
        data_parts = []
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_parts.append(line[len("data:") :].strip())
            elif line.startswith(":"):
                # SSE comment (e.g., heartbeat : ping)
                pass
        if event_name and data_parts:
            payload = json.loads("\n".join(data_parts))
            events.append({"event": event_name, "payload": payload})
    return events


class FakeGenerationService:
    """Mock service yielding deterministic generation events."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    async def generate(
        self, request: Any, request_id: str | None = None
    ) -> AsyncIterator[GenerationEvent]:
        req_id = request_id or "req_test01"
        yield GenerationEvent(
            type="meta",
            payload={
                "type": "meta",
                "request_id": req_id,
                "mode": "live",
                "provider": "gemini:gemini-3.1-flash-lite",
                "retrieval": {
                    "mode": "hybrid",
                    "k": 5,
                    "entities": ["PKG-120"],
                    "filters": {},
                    "gate": {"passed": True, "signal": "dense_cosine", "value": 0.85, "threshold": 0.45},
                    "latency_ms": 25.0,
                },
            },
        )

        if getattr(request, "debug", False):
            yield GenerationEvent(
                type="debug",
                payload={
                    "type": "debug",
                    "dense_ranks": [{"chunk_id": "PKG-120_CARD", "score": 0.85}],
                    "sparse_ranks": [{"chunk_id": "PKG-120_CARD", "score": 12.0}],
                    "fused_ranks": [{"chunk_id": "PKG-120_CARD", "rrf_score": 0.03}],
                },
            )

        if self.should_fail:
            yield GenerationEvent(
                type="error",
                payload={
                    "type": "error",
                    "code": "LLM_TIMEOUT",
                    "message": "Upstream LLM timed out.",
                    "retryable": True,
                },
            )
            yield GenerationEvent(
                type="done",
                payload={"type": "done", "status": "error", "latency_ms": 50.0, "ttft_ms": 0.0},
            )
            return

        yield GenerationEvent(
            type="token",
            payload={"type": "token", "content": "The CartonPro 1200 "},
        )
        yield GenerationEvent(
            type="token",
            payload={"type": "token", "content": "operates at 20 m/min [S1]."},
        )
        yield GenerationEvent(
            type="sources",
            payload={
                "type": "sources",
                "sources": [
                    {
                        "id": "S1",
                        "product_id": "PKG-120",
                        "product_name": "CartonPro 1200",
                        "document": "catalog.json",
                        "page": 1,
                        "chunk_id": "PKG-120_CARD",
                        "source_type": "structured",
                        "ocr_confidence": None,
                        "cited": True,
                        "suspicious": False,
                        "snippet": "CartonPro 1200 throughput 20 m/min",
                    }
                ],
            },
        )
        yield GenerationEvent(
            type="grounding",
            payload={"type": "grounding", "ok": True, "warnings": []},
        )
        yield GenerationEvent(
            type="done",
            payload={"type": "done", "status": "ok", "latency_ms": 120.0, "ttft_ms": 35.0},
        )


@pytest.fixture
def mock_app(test_settings: Settings):
    """Create test application with pre-injected mock dependencies."""
    app = create_app(custom_settings=test_settings)
    app.state.service = FakeGenerationService()

    mock_store = MagicMock()
    mock_store.count.return_value = 42
    mock_store.scroll_unique_values.side_effect = lambda field: {
        "category": ["Packaging", "Storage"],
        "supplier_name": ["PackRight"],
        "country": ["India", "Germany"],
        "product_id": ["PKG-120", "WHS-1800"],
    }.get(field, [])
    app.state.store = mock_store

    mock_router = MagicMock()
    provider_mock = MagicMock()
    provider_mock.name = "gemini"
    mock_router.providers = [provider_mock]
    app.state.router = mock_router

    return app


@pytest.mark.asyncio
async def test_sse_stream_full_contract(mock_app):
    """Verify SSE streaming emits meta -> token -> sources -> grounding -> done."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ask/stream",
            json={"question": "What is PKG-120?", "debug": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        events = parse_sse_events(resp.text)
        event_types = [e["event"] for e in events]

        # Check full event lifecycle
        assert event_types == ["meta", "debug", "token", "token", "sources", "grounding", "done"]

        # Check meta structure
        meta = events[0]["payload"]
        assert meta["type"] == "meta"
        assert meta["provider"] == "gemini:gemini-3.1-flash-lite"
        assert meta["retrieval"]["gate"]["passed"] is True

        # Check debug structure
        debug = events[1]["payload"]
        assert debug["type"] == "debug"
        assert len(debug["dense_ranks"]) > 0

        # Check tokens
        token1 = events[2]["payload"]
        assert token1["type"] == "token"
        assert "The CartonPro 1200" in token1["content"]

        # Check sources
        sources = events[4]["payload"]
        assert sources["type"] == "sources"
        assert sources["sources"][0]["id"] == "S1"

        # Check grounding
        grounding = events[5]["payload"]
        assert grounding["type"] == "grounding"
        assert grounding["ok"] is True

        # Check done
        done = events[6]["payload"]
        assert done["type"] == "done"
        assert done["status"] == "ok"
        assert done["latency_ms"] == 120.0


@pytest.mark.asyncio
async def test_sse_stream_error_flow(mock_app):
    """Verify failed stream emits error followed by done with status 'error'."""
    mock_app.state.service = FakeGenerationService(should_fail=True)
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ask/stream",
            json={"question": "Trigger error"},
        )
        assert resp.status_code == 200
        events = parse_sse_events(resp.text)
        event_types = [e["event"] for e in events]
        assert event_types == ["meta", "error", "done"]
        assert events[1]["payload"]["code"] == "LLM_TIMEOUT"
        assert events[2]["payload"]["status"] == "error"


@pytest.mark.asyncio
async def test_ask_sync_endpoint(mock_app):
    """Verify synchronous POST /ask buffers complete generation."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ask",
            json={"question": "What is PKG-120?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "The CartonPro 1200 operates at 20 m/min [S1]." in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["id"] == "S1"
        assert data["grounding"]["ok"] is True
        assert data["done"]["status"] == "ok"


@pytest.mark.asyncio
async def test_validation_empty_question(mock_app):
    """Verify empty question yields structured HTTP 422."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/ask", json={"question": ""})
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_validation_oversized_question(mock_app):
    """Verify question >1000 characters yields structured HTTP 422."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/ask", json={"question": "x" * 1001})
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_validation_out_of_range_params(mock_app):
    """Verify out-of-range top_k and threshold yield structured HTTP 422."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/ask", json={"question": "test", "top_k": 50})
        assert resp.status_code == 422

        resp2 = await client.post("/ask", json={"question": "test", "threshold": 1.5})
        assert resp2.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint_ok(mock_app):
    """Verify /health returns HTTP 200 with status 'ok' when LLM provider configured."""
    mock_app.state.settings.LLM_MODE = "live"
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["vector_db"]["reachable"] is True
        assert data["vector_db"]["collection_count"] == 42
        assert "gemini" in data["llm"]["active_providers"]


@pytest.mark.asyncio
async def test_health_endpoint_degraded(test_settings: Settings):
    """Verify /health returns status 'degraded' when no LLM keys configured."""
    degraded_settings = Settings(
        APP_ENV="test",
        QDRANT_MODE="memory",
        LLM_MODE="retrieval_only",
        GEMINI_API_KEY="",
        GROQ_API_KEY="",
        OPENROUTER_API_KEY="",
    )
    app = create_app(custom_settings=degraded_settings)
    mock_store = MagicMock()
    mock_store.count.return_value = 10
    app.state.store = mock_store
    app.state.router = MagicMock(providers=[])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"


@pytest.mark.asyncio
async def test_filters_endpoint(mock_app):
    """Verify /filters returns categories, suppliers, countries, and product_ids."""
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/filters")
        assert resp.status_code == 200
        data = resp.json()
        assert "Packaging" in data["categories"]
        assert "PackRight" in data["suppliers"]
        assert "India" in data["countries"]
        assert "PKG-120" in data["product_ids"]
