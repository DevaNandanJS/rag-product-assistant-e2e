"""Unit tests for configuration validation, fail-fast behavior, and secret masking."""

import pytest

from app.core.config import ConfigurationError, Settings


def test_config_defaults() -> None:
    s = Settings()
    assert s.APP_ENV in ("dev", "prod", "test")
    assert s.TOP_K == 5
    assert s.SCORE_THRESHOLD == 0.45
    assert s.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"


def test_config_secret_masking(test_settings: Settings) -> None:
    described = test_settings.describe()
    assert "AIzaSyDummyTestKey1234567890abcdefgh" not in str(described)
    assert "gsk_DummyGroqKey1234567890abcdefghijklmn" not in str(described)
    assert "***" in described["GEMINI_API_KEY"]
    assert "***" in described["GROQ_API_KEY"]


def test_config_invalid_qdrant_mode() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        Settings(QDRANT_MODE="invalid_mode")  # type: ignore
    assert "Invalid QDRANT_MODE" in str(exc_info.value)


def test_config_invalid_chunker() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        Settings(CHUNKER="unknown_chunker")  # type: ignore
    assert "Invalid CHUNKER" in str(exc_info.value)


def test_config_invalid_retrieval_mode() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        Settings(RETRIEVAL_MODE="magic")  # type: ignore
    assert "Invalid RETRIEVAL_MODE" in str(exc_info.value)


def test_config_invalid_llm_mode() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        Settings(LLM_MODE="hallucinate")  # type: ignore
    assert "Invalid LLM_MODE" in str(exc_info.value)
