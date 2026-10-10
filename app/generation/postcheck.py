"""Deterministic post-generation claim verifier and citation checker.

Extracts numeric/unit assertions, prices, and citation markers from generated
text, and validates them against the concatenated source context.
Outputs non-blocking GroundingWarning objects — never raises exceptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingWarning:
    """Represents a potentially ungrounded factual claim or invalid citation."""

    claim: str
    reason: str


# ---------------------------------------------------------------------------
# Extraction Patterns
# ---------------------------------------------------------------------------

# Numeric claims with physical / commercial units
_NUMERIC_UNIT_RE = re.compile(
    r"\b\d+[\.,]?\d*\s*(?:kW|V|AC|DC|kg|g|mm|cm|m|°C|m³|cartons/min|months|years|rpm|hours|days|Hz)\b",
    re.IGNORECASE,
)

# Currency and price figures
_PRICE_RE = re.compile(
    r"\b(?:USD|INR|Rs\.?|₹|\$|€)\s*[\d,]+(?:\.\d+)?\b",
    re.IGNORECASE,
)

# Citation markers [S1], [S2], etc.
_CITATION_RE = re.compile(r"\[S(\d+)\]")


# ---------------------------------------------------------------------------
# Claim Extraction
# ---------------------------------------------------------------------------


def extract_numeric_claims(text: str) -> list[str]:
    """Extract numeric/unit and price claims from text using regex.

    Returns deduplicated list of matched strings in order of first appearance.
    """
    if not text:
        return []

    claims: list[str] = []
    seen: set[str] = set()

    for pattern in (_PRICE_RE, _NUMERIC_UNIT_RE):
        for match in pattern.finditer(text):
            val = match.group().strip()
            if val.lower() not in seen:
                seen.add(val.lower())
                claims.append(val)

    return claims


def check_citations(generated: str, source_ids: set[str]) -> list[GroundingWarning]:
    """Validate that every [S#] citation marker in generated text exists in source_ids.

    Supports source_ids formatted as {"S1", "S2"} or {"1", "2"}.
    """
    if not generated:
        return []

    # Build canonical set of valid strings (both "1" and "S1")
    canonical_valid: set[str] = set()
    for s in source_ids:
        s_clean = s.strip()
        canonical_valid.add(s_clean.upper())
        if s_clean.upper().startswith("S"):
            canonical_valid.add(s_clean[1:])
        else:
            canonical_valid.add(f"S{s_clean.upper()}")

    warnings: list[GroundingWarning] = []
    for match in _CITATION_RE.finditer(generated):
        raw_num = match.group(1)
        full_tag = f"[S{raw_num}]"
        if raw_num not in canonical_valid and f"S{raw_num}" not in canonical_valid:
            warnings.append(
                GroundingWarning(
                    claim=full_tag,
                    reason=f"Citation {full_tag} does not match any retrieved source ID.",
                )
            )

    return warnings


# ---------------------------------------------------------------------------
# Main Verification Function
# ---------------------------------------------------------------------------


def verify(
    generated: str,
    context: str,
    source_ids: set[str],
) -> list[GroundingWarning]:
    """Deterministically audit generated text against retrieved context.

    1. Checks that all [S#] citation tags exist within source_ids.
    2. Checks that extracted numeric/unit specifications and prices exist
       within the concatenated context string.

    Returns a list of GroundingWarning objects. Returns empty list if all claims
    are grounded. Never raises exceptions.
    """
    if not generated:
        return []

    warnings: list[GroundingWarning] = []

    # 1. Audit citations
    warnings.extend(check_citations(generated, source_ids))

    # 2. Audit numeric claims against context
    claims = extract_numeric_claims(generated)
    if claims and context:
        norm_context = " ".join(context.lower().split())
        for claim in claims:
            norm_claim = " ".join(claim.lower().split())
            if norm_claim not in norm_context:
                warnings.append(
                    GroundingWarning(
                        claim=claim,
                        reason=f"Factual claim '{claim}' is not supported by retrieved context.",
                    )
                )

    return warnings
