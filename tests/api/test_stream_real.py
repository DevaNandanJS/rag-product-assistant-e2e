"""Integration test for True Streaming TTFT vs Total Latency verification.

Spins up a lightweight background Uvicorn server over a local loopback port
to assert that progressive SSE token streaming delivers Time-to-First-Token (TTFT)
significantly faster than total generation time over a real network socket.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest
import uvicorn
from httpx import AsyncClient

from app.api.app import create_app
from app.core.config import Settings
from app.generation.service import GenerationEvent


def get_free_port() -> int:
    """Find an available loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StreamingSimulationService:
    """Simulates an upstream provider yielding tokens progressively with realistic network latency."""

    async def generate(
        self, request: Any, request_id: str | None = None
    ) -> AsyncIterator[GenerationEvent]:
        t0 = time.perf_counter()
        yield GenerationEvent(
            type="meta",
            payload={
                "type": "meta",
                "request_id": request_id or "sim_req",
                "mode": "live",
                "provider": "simulation",
                "retrieval": {"gate": {"passed": True}},
            },
        )

        # First token arrives quickly (~25 ms)
        await asyncio.sleep(0.025)
        ttft = (time.perf_counter() - t0) * 1000
        yield GenerationEvent(
            type="token",
            payload={"type": "token", "content": "Token1 "},
        )

        # Subsequent tokens arrive sequentially across ~250 ms
        for i in range(2, 6):
            await asyncio.sleep(0.060)
            yield GenerationEvent(
                type="token",
                payload={"type": "token", "content": f"Token{i} "},
            )

        yield GenerationEvent(
            type="sources",
            payload={"type": "sources", "sources": []},
        )
        yield GenerationEvent(
            type="grounding",
            payload={"type": "grounding", "ok": True, "warnings": []},
        )
        total_latency = (time.perf_counter() - t0) * 1000
        yield GenerationEvent(
            type="done",
            payload={
                "type": "done",
                "status": "ok",
                "latency_ms": round(total_latency, 2),
                "ttft_ms": round(ttft, 2),
            },
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ttft_substantially_lower_than_total_latency(test_settings: Settings):
    """Verify true streaming: Time-to-First-Token (TTFT) is significantly lower than total generation time.

    Formula from Section 8 Study Notes:
        TTFT << Total Latency
    """
    app = create_app(custom_settings=test_settings)
    app.state.service = StreamingSimulationService()
    mock_store = MagicMock()
    mock_store.count.return_value = 1
    app.state.store = mock_store
    app.state.router = MagicMock(providers=[MagicMock(name="simulation")])

    port = get_free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for background Uvicorn server to start accepting connections
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.05)

    try:
        base_url = f"http://127.0.0.1:{port}"
        async with AsyncClient(base_url=base_url, timeout=10.0) as client:
            start_time = time.monotonic()
            first_token_time: float | None = None
            done_time: float | None = None

            async with client.stream(
                "POST",
                "/ask/stream",
                json={"question": "Simulate progressive stream"},
            ) as resp:
                assert resp.status_code == 200
                current_event = None
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("event:"):
                        current_event = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        if current_event == "token" and first_token_time is None:
                            first_token_time = time.monotonic()
                        elif current_event == "done":
                            done_time = time.monotonic()

            assert first_token_time is not None, "Did not receive any token events."
            assert done_time is not None, "Did not receive done event."

            measured_ttft_ms = (first_token_time - start_time) * 1000
            measured_total_ms = (done_time - start_time) * 1000

            # TTFT must be significantly smaller than total latency (e.g. less than 60% of total)
            assert measured_ttft_ms < (measured_total_ms * 0.6), (
                f"Fake streaming detected: TTFT ({measured_ttft_ms:.1f}ms) is not significantly "
                f"less than Total Latency ({measured_total_ms:.1f}ms)."
            )
    finally:
        server.should_exit = True
        server_thread.join(timeout=3.0)
