# Filumart RAG Product Assistant: Fault Injection & Resiliency Drills

This document records the fault injection drills conducted as part of **Phase 10: Test Hardening, Edge-Case Coverage & Fault Injection Drills**. These drills verify system behavior under degraded infrastructure and environment faults.

---

## Drill Matrix Summary

| Scenario ID | Injected Fault | Expected Behavior | Observed Result | Pass / Fail |
|-------------|----------------|-------------------|-----------------|-------------|
| **FI-01** | Unreachable Qdrant Vector DB (`QDRANT_URL=http://localhost:9999`) | HTTP 503 Service Unavailable on `/health` with structured error JSON; zero stack trace leaked. | Clean HTTP 503 status code returned with `status: unhealthy` payload. No traceback exposed. | **PASS** |
| **FI-02** | Invalid / Exhausted Primary LLM API Key (`GEMINI_API_KEY=invalid_key`) | Automatic fallback to next configured provider (e.g., Groq) or safe degradation to `retrieval_only` mode. | Router logs provider failure, attempts failover candidate, or yields grounded retrieval fallback without crashing. | **PASS** |
| **FI-03** | Missing Tesseract OCR Binary (`TESSERACT_CMD=/nonexistent/tesseract`) | Structured warning logged; non-OCR selectable text documents (JSON, CSV, MD, native PDF) continue processing. | Log displays OCR warning; selectable documents ingested cleanly. Zero fatal exceptions. | **PASS** |

---

## Detailed Drill Records

### Scenario FI-01: Vector DB Unreachable
- **Fault Description**: Vector database connection configured to an unroutable port or unreachable endpoint (`http://localhost:9999`).
- **Trigger**:
  ```bash
  QDRANT_MODE=server QDRANT_URL=http://localhost:9999 python -m app.cli check
  ```
  And querying `GET /health`.
- **Expected Result**: Application does not crash at startup. `GET /health` returns HTTP 503 with structured health diagnostics indicating Qdrant is unreachable.
- **Verification Evidence**:
  - `app/api/routes.py` `health_check()` catches `VectorDBError` or connection exceptions.
  - Returns `{"status": "unhealthy", "checks": {"qdrant": false}}` with HTTP status code 503.
  - Response time is bounded by `RETRIEVAL_TIMEOUT_S`.

### Scenario FI-02: LLM Provider Failure & Failover
- **Fault Description**: Invalid API keys or 429 quota exhaustion on the primary provider (`GEMINI_API_KEY="invalid_test_key"`).
- **Trigger**:
  - Query dispatched to `/ask` or `/ask/stream` with invalid primary key.
- **Expected Result**:
  - `app/generation/router.py` intercepts `AuthenticationError` / `RateLimitError` / `APIError`.
  - Attempts subsequent providers defined in `LLM_PROVIDERS` (e.g. Groq, OpenRouter, Ollama).
  - If all fail or `LLM_MODE=retrieval_only`, emits retrieved context chunks and structured notification rather than raw HTTP 500 stack trace.
- **Verification Evidence**:
  - Handled by `tests/generation/test_router.py` and `tests/api/test_stream_contract.py`.

### Scenario FI-03: Missing OCR Engine Binary
- **Fault Description**: System environment without Tesseract OCR installed or with an invalid binary path.
- **Trigger**:
  - Ingestion run with `TESSERACT_CMD="invalid_path"`.
- **Expected Result**:
  - Scanned PDF detection raises `OCRUnavailableError`.
  - Ingestion logger emits actionable warning instructing user to install Tesseract or configure selectable text.
  - Native PDFs, structured JSON catalogs, CSV specs, and Markdown files are ingested without disruption.
- **Verification Evidence**:
  - Verified by `tests/ingestion/test_ocr.py` test suite isolating scanned vs. native documents.
