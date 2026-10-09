"""Custom exception hierarchy for the Filumart RAG Product Assistant.
All application exceptions derive from AppError and provide clean error codes.
"""

from typing import Any


class AppError(Exception):
    """Base exception for all Filumart application errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class ConfigurationError(AppError):
    """Raised when environment or runtime settings fail validation."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="CONFIGURATION_ERROR",
            status_code=500,
            details=details,
        )


class IngestionError(AppError):
    """Raised when file parsing, chunking, or document preparation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="INGESTION_ERROR",
            status_code=422,
            details=details,
        )


class OCRUnavailableError(AppError):
    """Raised when OCR is requested but no local binary or vision model is available."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="OCR_UNAVAILABLE",
            status_code=503,
            details=details,
        )


class VectorDBError(AppError):
    """Raised when communication or operations with the vector store fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="VECTOR_DB_ERROR",
            status_code=502,
            details=details,
        )


class GateRejectionError(AppError):
    """Raised when a query fails the relevance threshold gate."""

    def __init__(
        self,
        message: str = "Query not sufficiently grounded in catalog knowledge base.",
        score: float | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="GATE_REJECTED",
            status_code=200,  # Graceful refusal rather than HTTP 5xx
            details={"score": score, "threshold": threshold},
        )


class LLMUnavailableError(AppError):
    """Raised when upstream language model providers are unreachable or unconfigured."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="LLM_UNAVAILABLE",
            status_code=503,
            details=details,
        )


class LLMTimeoutError(AppError):
    """Raised when upstream LLM response exceeds the first-token or idle deadline."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="LLM_TIMEOUT",
            status_code=504,
            details=details,
        )
