"""Structured JSON logging with credential redaction and contextual request tracing."""

import contextvars
import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

# Context variable for request tracing
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Secret patterns for redaction
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z-_]{35}"),  # Google API Key
    re.compile(r"gsk_[0-9A-Za-z]{30,}"),  # Groq API Key
    re.compile(r"sk-or-v1-[0-9A-Za-z]{40,}"),  # OpenRouter Key
    re.compile(r"sk-[0-9A-Za-z]{20,}"),  # Standard OpenAI-style Key
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I),  # Authorization Bearer tokens
]


def redact_secrets(text: str) -> str:
    """Replaces detected API keys or credentials in strings with [REDACTED]."""
    if not isinstance(text, str):
        return text
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as JSON lines, injecting request_id and redacting secrets."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
        }

        # Inject contextual request_id if present
        req_id = request_id_ctx.get()
        if req_id:
            log_entry["request_id"] = req_id

        # Attach exception details if any
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)

        # Include custom extra fields if provided
        for key, val in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "id",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                if isinstance(val, str):
                    log_entry[key] = redact_secrets(val)
                else:
                    log_entry[key] = val

        return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO", json_format: bool = True) -> logging.Logger:
    """Configures root logger with redaction and structured formatting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        stream_handler.setFormatter(StructuredJsonFormatter())
    else:
        stream_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(stream_handler)
    return logging.getLogger("filumart")


logger = setup_logging()
