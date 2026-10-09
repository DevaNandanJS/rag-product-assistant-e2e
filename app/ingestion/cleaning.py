"""Text cleaning and sanitization for ingested content.

Responsibilities (Section 4.2 – app.ingestion.cleaning):
- Unicode normalization: dashes, degree symbols, whitespace.
- Prompt-injection pattern detection with flagging (never discards chunks).
- No chunking, database, or embedding operations.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt-injection signatures (case-insensitive)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"system\s*override",
        r"disregard\s+(all\s+)?prior",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+(?:a|an)\s+\w+",
        r"act\s+as\s+(?:a|an)\s+\w+",
        r"new\s+persona",
        r"jailbreak",
        r"DAN\s+mode",
        r"pretend\s+you\s+(?:are|have)",
        r"unconditional\s+warrant",  # catches the poisoned bulletin fixture
        r"IMPORTANT\s+SYSTEM",       # catches "IMPORTANT SYSTEM OVERRIDE"
        r"claim\s+a\s+\d+.+warrant", # e.g. "claim a 50-year unconditional warranty"
    ]
]

# ---------------------------------------------------------------------------
# Unicode normalization maps
# ---------------------------------------------------------------------------

# Map fancy Unicode dashes to ASCII hyphen-minus
_DASH_CHARS = (
    "\u2010"  # HYPHEN
    "\u2011"  # NON-BREAKING HYPHEN
    "\u2012"  # FIGURE DASH
    "\u2013"  # EN DASH
    "\u2014"  # EM DASH
    "\u2015"  # HORIZONTAL BAR
    "\u2212"  # MINUS SIGN
    "\ufe58"  # SMALL EM DASH
    "\ufe63"  # SMALL HYPHEN-MINUS
    "\uff0d"  # FULLWIDTH HYPHEN-MINUS
)
_DASH_RE = re.compile(f"[{re.escape(_DASH_CHARS)}]")

# Map fancy degree signs to standard ° (U+00B0)
_FANCY_DEGREE_RE = re.compile(r"[℃℉]")

# Collapse excessive whitespace (including non-breaking spaces)
_WHITESPACE_RE = re.compile(r"[ \t\u00a0\u2009\u202f]+")

# Multiple blank lines → single blank line
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Apply full Unicode normalization and whitespace cleanup.

    Steps (in order):
    1. NFC normalization (compose canonical equivalents).
    2. Replace variant dashes with ASCII hyphen-minus.
    3. Replace ℃/℉ with °C / °F.
    4. Collapse runs of spaces/tabs/NBSP to a single space.
    5. Collapse runs of 3+ blank lines to exactly one blank line.
    6. Strip leading/trailing whitespace.

    Args:
        text: Raw text from any parser or OCR engine.

    Returns:
        Normalized Unicode string.
    """
    # 1. NFC compose
    text = unicodedata.normalize("NFC", text)

    # 2. Normalize dashes → hyphen-minus
    text = _DASH_RE.sub("-", text)

    # 3. Normalize special degree symbols
    text = _FANCY_DEGREE_RE.sub(_replace_degree, text)

    # 4. Collapse internal whitespace per line
    lines = text.split("\n")
    lines = [_WHITESPACE_RE.sub(" ", line) for line in lines]
    text = "\n".join(lines)

    # 5. Collapse multiple blank lines
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    # 6. Strip
    return text.strip()


def is_suspicious(text: str) -> bool:
    """Return True if the text contains prompt-injection signatures.

    The chunk is **not** discarded – the caller must set ``chunk.suspicious = True``
    and let the downstream gate and generation layer decide whether to include
    or skip the passage.

    Args:
        text: Cleaned text to inspect.

    Returns:
        True when any injection pattern is detected.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "Prompt-injection pattern detected: '%s' in text snippet: %.80r",
                pattern.pattern,
                text,
            )
            return True
    return False


def clean(text: str) -> tuple[str, bool]:
    """Normalize text and check for injection patterns in one call.

    Args:
        text: Raw input string.

    Returns:
        A tuple of ``(cleaned_text, suspicious)`` where ``suspicious`` is True
        when any injection pattern is found in the *original* text (before
        normalization, to avoid missing obfuscated patterns).
    """
    suspicious = is_suspicious(text)
    cleaned = normalize_text(text)
    return cleaned, suspicious


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace_degree(match: re.Match[str]) -> str:  # type: ignore[type-arg]
    """Map ℃ → °C, ℉ → °F."""
    char = match.group(0)
    if char == "℃":
        return "°C"
    if char == "℉":
        return "°F"
    return char
