"""Unit tests for the application exception hierarchy and error formatting."""

from app.core.errors import (
    AppError,
    ConfigurationError,
    GateRejectionError,
    IngestionError,
    LLMTimeoutError,
    LLMUnavailableError,
    OCRUnavailableError,
    VectorDBError,
)


def test_app_error_dict() -> None:
    err = AppError(
        "Sample internal failure",
        code="TEST_CODE",
        status_code=500,
        details={"k": "v"},
    )
    d = err.to_dict()
    assert d["error"]["code"] == "TEST_CODE"
    assert d["error"]["message"] == "Sample internal failure"
    assert d["error"]["details"]["k"] == "v"


def test_specific_exceptions() -> None:
    cfg_err = ConfigurationError("Bad config")
    assert cfg_err.code == "CONFIGURATION_ERROR"
    assert cfg_err.status_code == 500

    ing_err = IngestionError("Corrupt document")
    assert ing_err.code == "INGESTION_ERROR"
    assert ing_err.status_code == 422

    ocr_err = OCRUnavailableError("Tesseract missing")
    assert ocr_err.code == "OCR_UNAVAILABLE"
    assert ocr_err.status_code == 503

    vdb_err = VectorDBError("Qdrant unreachable")
    assert vdb_err.code == "VECTOR_DB_ERROR"
    assert vdb_err.status_code == 502

    gate_err = GateRejectionError(score=0.21, threshold=0.45)
    assert gate_err.code == "GATE_REJECTED"
    assert gate_err.status_code == 200
    assert gate_err.details["score"] == 0.21

    llm_unavail = LLMUnavailableError("No active provider")
    assert llm_unavail.code == "LLM_UNAVAILABLE"
    assert llm_unavail.status_code == 503

    llm_to = LLMTimeoutError("First token timed out")
    assert llm_to.code == "LLM_TIMEOUT"
    assert llm_to.status_code == 504
