"""Pytest shared fixtures and offline testing setup.
Enforces offline in-memory settings for unit tests.
"""

import pytest

from app.core.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Provides a deterministic test settings object in memory mode."""
    return Settings(
        APP_ENV="test",
        LOG_LEVEL="DEBUG",
        QDRANT_MODE="memory",
        CHUNKER="structured",
        LLM_MODE="retrieval_only",
        GEMINI_API_KEY="AIzaSyDummyTestKey1234567890abcdefgh",
        GROQ_API_KEY="gsk_DummyGroqKey1234567890abcdefghijklmn",
    )
