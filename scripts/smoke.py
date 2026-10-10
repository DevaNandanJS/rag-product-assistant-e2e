#!/usr/bin/env python3
"""Smoke test script for verifying containerized or live Filumart RAG service."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def poll_health(base_url: str, max_wait_s: float = 60.0, poll_interval_s: float = 2.0) -> dict:
    """Poll the /health endpoint until status is 'ok' or 'degraded'."""
    health_url = f"{base_url.rstrip('/')}/health"
    start_time = time.time()
    print(f"[SMOKE] Polling {health_url} (max {max_wait_s}s)...")

    last_error = None
    while time.time() - start_time < max_wait_s:
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "smoke-test/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    status = data.get("status")
                    print(f"[SMOKE] /health responded status='{status}' (code {resp.status})")
                    if status in ("ok", "degraded"):
                        return data
        except urllib.error.HTTPError as he:
            # 503 degraded can still return JSON during startup
            try:
                body = json.loads(he.read().decode("utf-8"))
                if body.get("status") in ("ok", "degraded"):
                    print(f"[SMOKE] /health returned {body.get('status')} with code {he.code}")
                    return body
            except Exception:
                pass
            last_error = f"HTTP {he.code}: {he.reason}"
        except Exception as e:
            last_error = str(e)

        time.sleep(poll_interval_s)

    raise TimeoutError(f"Service at {health_url} failed to become healthy within {max_wait_s}s. Last error: {last_error}")


def test_ask_stream(base_url: str, question: str = "What is the CartonPro 1200?") -> None:
    """Post query to /ask/stream and verify stream terminates with 'done'."""
    stream_url = f"{base_url.rstrip('/')}/ask/stream"
    payload = json.dumps({"question": question, "debug": True}).encode("utf-8")
    req = urllib.request.Request(
        stream_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "keep-alive",
        },
        method="POST",
    )

    print(f"[SMOKE] Sending test query to {stream_url}: '{question}'...")
    events_received: list[str] = []
    has_done_event = False

    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Unexpected status code {resp.status} from /ask/stream")

        current_event = None
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("event:"):
                current_event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                event_name = current_event or "message"
                events_received.append(event_name)
                if event_name == "done":
                    has_done_event = True
                current_event = None
            elif not line:
                if current_event:
                    events_received.append(current_event)
                    if current_event == "done":
                        has_done_event = True
                    current_event = None

    print(f"[SMOKE] Received SSE events: {events_received}")
    if not has_done_event:
        raise AssertionError(f"SSE stream did not complete with 'done' event. Events: {events_received}")
    print("[SMOKE] Stream completed successfully with 'done' event.")


def main() -> int:
    base_url = os.environ.get("BASE_URL", sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
    try:
        health_data = poll_health(base_url)
        print(f"[OK] Health check passed: {health_data}")
        test_ask_stream(base_url)
        print("[SUCCESS] Smoke test passed completely!")
        return 0
    except Exception as exc:
        print(f"[FAILURE] Smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
