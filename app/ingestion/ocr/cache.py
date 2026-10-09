"""SHA-256-keyed disk cache for vision OCR responses.

Cache entries are stored as JSONL records in the file specified by
``Settings.VISION_CACHE_PATH``. Each record has the shape:

    {"key": "<sha256>", "text": "<transcription>", "model": "<model_id>"}

Responsibilities (Section 4.2):
- Read and write cache entries atomically.
- No OCR logic, no LLM calls, no vector-store access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _make_key(image_bytes: bytes, prompt: str, model: str) -> str:
    """Compute a deterministic SHA-256 key from image bytes + prompt + model."""
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(prompt.encode("utf-8"))
    h.update(model.encode("utf-8"))
    return h.hexdigest()


class VisionOCRCache:
    """Append-only JSONL cache for vision-model OCR transcriptions.

    The cache is loaded once at construction time into an in-memory dict;
    subsequent lookups are O(1). Writes are appended to the backing file
    without rewriting the entire file.
    """

    def __init__(self, cache_path: str | Path) -> None:
        self._path = Path(cache_path)
        self._store: dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, image_bytes: bytes, prompt: str, model: str) -> str | None:
        """Return cached transcription or *None* on cache-miss."""
        key = _make_key(image_bytes, prompt, model)
        return self._store.get(key)

    def put(self, image_bytes: bytes, prompt: str, model: str, text: str) -> None:
        """Store a new transcription and flush the record to disk."""
        key = _make_key(image_bytes, prompt, model)
        if key in self._store:
            return  # already cached – avoid duplicating records

        self._store[key] = text
        self._append({"key": key, "text": text, "model": model})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Populate in-memory store from the JSONL file (if it exists)."""
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.get("key", "")
                    text = record.get("text", "")
                    if key:
                        self._store[key] = text
                except json.JSONDecodeError:
                    # Corrupt line – skip gracefully
                    continue

    def _append(self, record: dict) -> None:
        """Append a JSON record to the JSONL file, creating parent dirs as needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
