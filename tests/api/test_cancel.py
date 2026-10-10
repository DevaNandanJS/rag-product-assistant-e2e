"""Tests for async SSE client disconnect and upstream generator cancellation."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.sse import sse_generator
from app.core.schemas import AskRequest
from app.generation.service import GenerationEvent


@pytest.mark.asyncio
async def test_sse_generator_aborts_on_disconnect():
    """Verify that when request.is_disconnected() returns True, upstream generator is cleanly closed within 500 ms."""
    generator_closed = asyncio.Event()

    async def slow_generator(req, request_id=None):
        try:
            yield GenerationEvent(type="meta", payload={"type": "meta"})
            yield GenerationEvent(type="token", payload={"type": "token", "content": "first "})
            for i in range(100):
                await asyncio.sleep(0.05)
                yield GenerationEvent(
                    type="token", payload={"type": "token", "content": f"token_{i} "}
                )
        except (GeneratorExit, asyncio.CancelledError):
            pass
        finally:
            generator_closed.set()

    mock_service = MagicMock()
    mock_service.generate.side_effect = slow_generator

    call_count = 0

    async def mock_is_disconnected():
        nonlocal call_count
        call_count += 1
        return call_count >= 2  # Disconnect after second iteration check

    mock_request = MagicMock()
    mock_request.is_disconnected = AsyncMock(side_effect=mock_is_disconnected)

    ask_req = AskRequest(question="Test cancellation")
    sse_gen = sse_generator(
        request=mock_request,
        service=mock_service,
        ask_request=ask_req,
        request_id="req_cancel_test",
        heartbeat_s=15.0,
    )

    t0 = time.monotonic()
    events = []
    async for item in sse_gen:
        events.append(item)

    elapsed_ms = (time.monotonic() - t0) * 1000

    # Upstream generator must be closed cleanly via finally block
    assert generator_closed.is_set(), "Upstream generator finally block was not executed."
    # Abort must occur well within 500 ms
    assert elapsed_ms < 500.0, f"Generator abort took {elapsed_ms:.1f}ms, exceeding 500ms limit."
    # Only the initial items before disconnect should be yielded
    assert len(events) <= 3


@pytest.mark.asyncio
async def test_sse_generator_aclose_propagation():
    """Verify that calling aclose() directly on sse_generator terminates upstream generator."""
    generator_closed = asyncio.Event()

    async def long_running_generator(req, request_id=None):
        try:
            yield GenerationEvent(type="meta", payload={"type": "meta"})
            while True:
                await asyncio.sleep(0.1)
                yield GenerationEvent(type="token", payload={"type": "token", "content": "tok "})
        except (GeneratorExit, asyncio.CancelledError):
            pass
        finally:
            generator_closed.set()

    mock_service = MagicMock()
    mock_service.generate.side_effect = long_running_generator

    mock_request = MagicMock()
    mock_request.is_disconnected = AsyncMock(return_value=False)

    ask_req = AskRequest(question="Test manual close")
    sse_gen = sse_generator(
        request=mock_request,
        service=mock_service,
        ask_request=ask_req,
        request_id="req_manual_close",
        heartbeat_s=15.0,
    )

    # Consume first event then close
    first_event = await sse_gen.asend(None)
    assert first_event.event == "meta"

    t0 = time.monotonic()
    await sse_gen.aclose()
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert generator_closed.is_set(), "Upstream generator was not closed by aclose()."
    assert elapsed_ms < 500.0, f"aclose() took {elapsed_ms:.1f}ms, exceeding 500ms limit."
