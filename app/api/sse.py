"""Server-Sent Events (SSE) streaming generator and formatting.

Provides sse_generator wrapping GenerationService event streams with:
- Section 5.4 SSE event frames (meta, debug, token, sources, grounding, done, error)
- Client disconnect detection and upstream cancellation
- Heartbeat ping comments (: ping)
- Safe error handling without raw traceback leakage
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import Request
from sse_starlette.sse import ServerSentEvent

from app.core.schemas import AskRequest
from app.generation.service import GenerationService

logger = logging.getLogger("app.api.sse")


async def sse_generator(
    request: Request,
    service: GenerationService,
    ask_request: AskRequest,
    request_id: str,
    heartbeat_s: float = 15.0,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Wrap GenerationService.generate() into sse-starlette ServerSentEvent frames.

    Args:
        request: FastAPI HTTP request for checking client disconnects.
        service: GenerationService orchestrating retrieval + router.
        ask_request: Validated user query payload.
        request_id: Unique correlation ID for logging and telemetry.
        heartbeat_s: Seconds of inactivity before emitting a ': ping' comment.

    Yields:
        ServerSentEvent frames.
    """
    gen = service.generate(ask_request, request_id=request_id)
    last_sent = time.monotonic()

    try:
        async for event in gen:
            # Check client disconnect without blocking normal iteration
            try:
                if callable(getattr(request, "is_disconnected", None)):
                    disc_fut = request.is_disconnected()
                    if asyncio.iscoroutine(disc_fut):
                        is_disc = await asyncio.wait_for(disc_fut, timeout=0.0001)
                    else:
                        is_disc = bool(disc_fut)
                    if is_disc:
                        logger.info(
                            "[sse] Client disconnected for request %s; aborting generation.",
                            request_id,
                        )
                        await gen.aclose()
                        return
            except (asyncio.TimeoutError, TimeoutError):
                pass
            except Exception:
                pass

            now = time.monotonic()
            if now - last_sent >= heartbeat_s:
                yield ServerSentEvent(comment="ping")
                last_sent = time.monotonic()

            yield ServerSentEvent(
                event=event.type,
                data=json.dumps(event.payload),
            )
            last_sent = time.monotonic()

    except asyncio.CancelledError:
        logger.info("[sse] Generation stream cancelled for request %s.", request_id)
        await gen.aclose()
        raise
    except Exception as exc:
        logger.exception(
            "[sse] Unexpected error in SSE stream for request %s: %s",
            request_id,
            exc,
        )
        yield ServerSentEvent(
            event="error",
            data=json.dumps(
                {
                    "type": "error",
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                    "retryable": False,
                }
            ),
        )
        yield ServerSentEvent(
            event="done",
            data=json.dumps(
                {
                    "type": "done",
                    "status": "error",
                    "latency_ms": 0.0,
                    "ttft_ms": 0.0,
                }
            ),
        )
    finally:
        try:
            await gen.aclose()
        except Exception:
            pass
