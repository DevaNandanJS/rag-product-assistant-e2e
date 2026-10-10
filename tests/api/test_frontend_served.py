"""Tests for static web interface serving, asset availability, and XSS safety."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.core.config import Settings
from app.generation.service import GenerationEvent


class DummyGenerationService:
    """Lightweight dummy service for API endpoint handling."""

    async def generate(self, request, request_id=None):
        yield GenerationEvent(
            type="meta",
            payload={"type": "meta", "request_id": request_id or "test-req"},
        )
        yield GenerationEvent(
            type="token",
            payload={"type": "token", "token": "Safe answer text"},
        )
        yield GenerationEvent(
            type="done",
            payload={"type": "done", "status": "completed", "latency_ms": 10.0},
        )


@pytest.fixture
def test_app(test_settings: Settings):
    app = create_app(custom_settings=test_settings)
    app.state.service = DummyGenerationService()
    return app


@pytest.mark.asyncio
async def test_index_html_served(test_app):
    """Assert GET / serves the index.html page with correct semantic elements."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        html = resp.text
        assert "Filumart" in html
        assert "<!DOCTYPE html>" in html
        assert 'id="query-input"' in html
        assert 'id="btn-send"' in html
        assert 'id="btn-stop"' in html
        assert 'id="debug-drawer"' in html
        assert 'id="status-badge"' in html


@pytest.mark.asyncio
async def test_static_assets_served(test_app):
    """Assert all core frontend CSS, JS, and vendor assets are served correctly."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        # Stylesheet
        resp = await client.get("/styles.css")
        assert resp.status_code == 200
        assert ":root" in resp.text

        # API module
        resp = await client.get("/api.js")
        assert resp.status_code == 200
        assert "streamAsk" in resp.text

        # Render module
        resp = await client.get("/render.js")
        assert resp.status_code == 200
        assert "renderMarkdown" in resp.text

        # App module
        resp = await client.get("/app.js")
        assert resp.status_code == 200
        assert "setupEventListeners" in resp.text

        # Vendor README
        resp = await client.get("/vendor/README.md")
        assert resp.status_code == 200
        assert "marked.min.js" in resp.text

        # Pinned vendor JS files
        resp_marked = await client.get("/vendor/marked.min.js")
        assert resp_marked.status_code == 200
        assert len(resp_marked.text) > 1000

        resp_purify = await client.get("/vendor/purify.min.js")
        assert resp_purify.status_code == 200
        assert len(resp_purify.text) > 1000


@pytest.mark.asyncio
async def test_xss_query_handling(test_app):
    """Assert malicious XSS strings are accepted safely by the API without server crash."""
    xss_payload = "<img src=x onerror=alert('xss')><script>alert(1)</script>"
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/ask/stream",
            json={"question": xss_payload, "debug": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        # Assert streaming events completed without 500 error
        assert "event: done" in body
