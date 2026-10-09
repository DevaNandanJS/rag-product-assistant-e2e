"""Unit tests for structured logging and secret redaction."""

import json
import logging

from app.core.logging import StructuredJsonFormatter, redact_secrets, request_id_ctx


def test_redact_secrets() -> None:
    text = (
        "Key is AIzaSyD00000000000000000000000000000000 "
        "and groq is gsk_111111111111111111111111111111"
    )
    sanitized = redact_secrets(text)
    assert "AIza" not in sanitized
    assert "gsk_" not in sanitized
    assert "[REDACTED]" in sanitized


def test_structured_json_formatter() -> None:
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="API call failed with Bearer secret_token_xyz_12345",
        args=(),
        exc_info=None,
    )

    request_id_ctx.set("req_test_123")
    try:
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["logger"] == "test_logger"
        assert parsed["level"] == "INFO"
        assert parsed["request_id"] == "req_test_123"
        assert "secret_token_xyz" not in parsed["message"]
        assert "[REDACTED]" in parsed["message"]
    finally:
        request_id_ctx.set(None)
