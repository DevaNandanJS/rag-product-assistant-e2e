"""JSONL response cache for generated LLM answers.

Provides SHA256-keyed write-through caching of full LLM completions to
prevent duplicate API billing during testing and evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GenerationCache:
    """Append-only JSONL generation cache keyed by SHA256(messages + model)."""

    def __init__(self, cache_path: str | Path, enabled: bool = False) -> None:
        self.cache_path = Path(cache_path)
        self.enabled = enabled
        self._entries: dict[str, str] | None = None

    def _ensure_loaded(self) -> None:
        """Load cached responses from the JSONL file into memory on first access."""
        if self._entries is not None:
            return

        self._entries = {}
        if not self.cache_path.exists():
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        k = record.get("key")
                        v = record.get("response")
                        if k and v is not None:
                            self._entries[k] = v
                    except json.JSONDecodeError as err:
                        logger.warning(
                            "[cache] Skipping corrupt record at %s:%d: %s",
                            self.cache_path,
                            line_num,
                            err,
                        )
            logger.info(
                "[cache] Loaded %d cached responses from %s",
                len(self._entries),
                self.cache_path,
            )
        except OSError as exc:
            logger.warning("[cache] Failed to read cache file %s: %s", self.cache_path, exc)

    @staticmethod
    def make_key(messages: list[dict[str, Any]], model: str) -> str:
        """Derive a deterministic SHA256 cache key from messages and model name."""
        serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        payload = f"{serialized}:{model}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def lookup(self, key: str) -> str | None:
        """Return the cached response string for key, or None if not cached."""
        self._ensure_loaded()
        assert self._entries is not None
        return self._entries.get(key)

    def write(self, key: str, response: str) -> None:
        """Persist a response to the in-memory index and append to the JSONL file."""
        if not self.enabled:
            return

        self._ensure_loaded()
        assert self._entries is not None

        if key in self._entries:
            return  # Idempotent: already written

        self._entries[key] = response

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "key": key,
                "response": response,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(
                "[cache] Failed to append entry to %s: %s", self.cache_path, exc
            )
