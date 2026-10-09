"""Configuration management for Filumart RAG Assistant.
All settings are parsed and validated via pydantic-settings from environment and .env files.
"""

from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import ConfigurationError


class Settings(BaseSettings):
    """Central configuration class declaring all runtime, ingestion,
    retrieval, and LLM parameters.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 1. Environment & Logging
    APP_ENV: Literal["dev", "prod", "test"] = "dev"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)

    # 2. Vector DB (Qdrant)
    QDRANT_MODE: Literal["server", "local", "memory"] = "server"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_PATH: str = "./qdrant_local"
    COLLECTION_PREFIX: str = "filumart"

    # 3. Ingestion & Chunking
    CHUNKER: Literal["structured", "fixed"] = "structured"
    CHUNK_TARGET_TOKENS: int = Field(default=250, ge=50, le=1000)
    DATA_DIR: str = "data/raw"

    # 4. Embeddings & Reranking
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    FASTEMBED_CACHE_PATH: str = "./.fastembed_cache"
    RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-12-v2"

    # 5. Retrieval & Gating
    RETRIEVAL_MODE: Literal["hybrid", "dense", "sparse"] = "hybrid"
    TOP_K: int = Field(default=5, ge=1, le=20)
    CANDIDATES: int = Field(default=20, ge=1, le=50)
    RRF_K: int = Field(default=60, ge=1)
    ENTITY_BOOST: float = Field(default=1.25, ge=1.0, le=3.0)
    PER_PRODUCT_CAP: int = Field(default=3, ge=1, le=10)
    GATE_ENABLED: bool = True
    SCORE_THRESHOLD: float = Field(default=0.45, ge=0.0, le=1.0)

    # 6. LLM Providers & Credentials
    LLM_PROVIDERS: str = "gemini,groq,openrouter,ollama"
    GEMINI_API_KEY: str = ""
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"

    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    JUDGE_MODEL: str = "llama-3.3-70b-versatile"

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"

    OLLAMA_BASE_URL: str = ""
    OLLAMA_MODEL: str = "qwen2.5:3b"

    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=700, ge=50, le=4096)
    LLM_MODE: Literal["live", "retrieval_only", "replay"] = "live"
    LLM_CACHE_PATH: str = "data/cache/llm_answers.jsonl"
    LLM_CACHE_WRITE: bool = False

    # 7. Timeouts & Keep-Alives
    EMBED_TIMEOUT_S: float = Field(default=10.0, gt=0.0)
    RETRIEVAL_TIMEOUT_S: float = Field(default=10.0, gt=0.0)
    LLM_FIRST_TOKEN_TIMEOUT_S: float = Field(default=20.0, gt=0.0)
    LLM_IDLE_TIMEOUT_S: float = Field(default=20.0, gt=0.0)
    REQUEST_TIMEOUT_S: float = Field(default=90.0, gt=0.0)
    HEARTBEAT_S: float = Field(default=15.0, gt=0.0)

    # 8. OCR & Image Preprocessing
    TESSERACT_CMD: str = ""
    OCR_LANG: str = "eng"
    OCR_DPI: int = Field(default=300, ge=72, le=600)
    OCR_MIN_TEXT_CHARS: int = Field(default=50, ge=10)
    OCR_CONF_THRESHOLD: float = Field(default=0.60, ge=0.0, le=1.0)
    VISION_FALLBACK: bool = True
    VISION_CACHE_PATH: str = "data/cache/vision_ocr.jsonl"

    @field_validator("QDRANT_MODE", mode="before")
    @classmethod
    def validate_qdrant_mode(cls, v: Any) -> Any:
        valid_modes = {"server", "local", "memory"}
        if str(v).lower() not in valid_modes:
            raise ConfigurationError(
                f"Invalid QDRANT_MODE: '{v}'. Must be one of: {sorted(valid_modes)}"
            )
        return str(v).lower()

    @field_validator("RETRIEVAL_MODE", mode="before")
    @classmethod
    def validate_retrieval_mode(cls, v: Any) -> Any:
        valid_modes = {"hybrid", "dense", "sparse"}
        if str(v).lower() not in valid_modes:
            raise ConfigurationError(
                f"Invalid RETRIEVAL_MODE: '{v}'. Must be one of: {sorted(valid_modes)}"
            )
        return str(v).lower()

    @field_validator("CHUNKER", mode="before")
    @classmethod
    def validate_chunker(cls, v: Any) -> Any:
        valid_chunkers = {"structured", "fixed"}
        if str(v).lower() not in valid_chunkers:
            raise ConfigurationError(
                f"Invalid CHUNKER: '{v}'. Must be one of: {sorted(valid_chunkers)}"
            )
        return str(v).lower()

    @field_validator("LLM_MODE", mode="before")
    @classmethod
    def validate_llm_mode(cls, v: Any) -> Any:
        valid_modes = {"live", "retrieval_only", "replay"}
        if str(v).lower() not in valid_modes:
            raise ConfigurationError(
                f"Invalid LLM_MODE: '{v}'. Must be one of: {sorted(valid_modes)}"
            )
        return str(v).lower()

    def describe(self) -> dict[str, Any]:
        """Returns a sanitized dict of configuration parameters with masked credentials."""
        secret_keys = {
            "GEMINI_API_KEY",
            "GROQ_API_KEY",
            "OPENROUTER_API_KEY",
        }
        res: dict[str, Any] = {}
        for key, val in self.model_dump().items():
            if key in secret_keys:
                placeholders = {
                    "",
                    "your_key_here",
                    "your_gemini_api_key_here",
                    "your_groq_api_key_here",
                }
                if not val or val in placeholders:
                    res[key] = "<not set>"
                else:
                    visible_len = min(4, len(val))
                    res[key] = f"{val[:visible_len]}...*** (len={len(val)})"
            else:
                res[key] = val
        return res

    def get_providers_list(self) -> list[str]:
        """Returns list of enabled LLM providers parsed from LLM_PROVIDERS."""
        return [p.strip().lower() for p in self.LLM_PROVIDERS.split(",") if p.strip()]


def get_settings() -> Settings:
    """Factory helper to obtain a validated Settings instance."""
    try:
        return Settings()
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Failed to load application settings: {exc}") from exc


settings = get_settings()
