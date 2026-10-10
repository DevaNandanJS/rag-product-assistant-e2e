"""Canonical data schemas, contracts, and Pydantic models for Filumart RAG Assistant.
Covers domain models, chunk contracts, API requests, and response models.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# 1. Canonical Domain Models (Multi-Category B2B Catalogs)
# ---------------------------------------------------------------------------


class SpecValue(BaseModel):
    """Specification value with optional unit and raw textual representation."""

    model_config = ConfigDict(frozen=True)

    value: str | float | int | None = None  # None explicitly represents "Not documented"
    unit: str | None = None
    raw: str | None = None  # Original textual representation


class ProductRecord(BaseModel):
    """Canonical model for products ingested into the catalog."""

    product_id: str  # e.g., "PKG-120", "WHS-1800", "REF-320", "TEX-12"
    product_name: str  # e.g., "CartonPro 1200 Semi-Automatic Carton Sealer"
    category: str  # e.g., "packaging_equipment", "warehouse_storage", etc.
    supplier_id: str | None = None  # e.g., "SUP-PKG-11"
    supplier_name: str | None = None  # e.g., "PackRight Systems Pvt. Ltd."
    country: str | None = "IN"  # ISO-2 Country code
    description: str | None = None
    specs: dict[str, SpecValue] = Field(default_factory=dict)
    # Common keys: capacity, power, operating_temperature, dimensions, warranty, price
    source_document: str  # e.g., "catalog.json", "packaging_bulletins.pdf"


class SupplierRecord(BaseModel):
    """Canonical model for supplier entities."""

    supplier_id: str  # e.g., "SUP-PKG-11"
    supplier_name: str
    headquarters: str  # e.g., "Pune, Maharashtra, India"
    country: str = "IN"
    categories: list[str] = Field(default_factory=list)
    regions_served: list[str] = Field(default_factory=list)
    lead_time_notes: str | None = None
    source_document: str


class TechnicalBulletin(BaseModel):
    """Canonical model for technical notes, safety bulletins, and advisories."""

    bulletin_id: str  # e.g., "PR-TECH-07", "WHS-SAFE-04"
    title: str
    related_product_ids: list[str] = Field(default_factory=list)
    content: str
    source_document: str


# ---------------------------------------------------------------------------
# 2. Chunk & Vector Storage Contracts
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """Normalized chunk model stored in Qdrant and passed across retrieval/generation."""

    chunk_id: str  # e.g., "PKG-120_CARD", "WHS-1800_SPEC_01"
    point_id: str  # UUIDv5 string derived from schema+document+chunk_id
    text: str  # Context header + body (sent to Embedder)
    display_text: str  # Body text without headers (sent to LLM context & UI)
    chunk_type: Literal["card", "spec", "prose", "ocr_page", "ocr_block", "fixed"]
    product_id: str | None = None
    product_name: str | None = None
    category: str | None = None
    supplier_id: str | None = None
    supplier_name: str | None = None
    country: str | None = None
    document: str  # Original filename
    page: int | None = None  # 1-indexed page number
    section: str | None = None
    source_type: Literal["structured", "extracted", "ocr", "ocr_vision"]
    ocr_confidence: float | None = None  # Scaled 0.0 to 1.0
    token_count: int
    suspicious: bool = False  # Flagged for prompt injection patterns
    schema_version: int = 1


# ---------------------------------------------------------------------------
# 3. Request & Response Contracts (HTTP & SSE)
# ---------------------------------------------------------------------------


class Filters(BaseModel):
    """Optional metadata filters applied during vector and sparse retrieval."""

    category: str | None = None
    country: str | None = None
    supplier_name: str | None = None
    product_id: str | None = None


class AskRequest(BaseModel):
    """Request payload for /ask and /ask/stream endpoints.

    Accepts both 'question' and 'query' field names for seamless compatibility
    with the assessment specification (Section 10).
    """

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(..., min_length=1, max_length=1000)
    filters: Filters | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    history: list[dict[str, str]] | None = Field(default=None, max_length=6)
    debug: bool = False

    @model_validator(mode="before")
    @classmethod
    def populate_question_from_query(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("question") and data.get("query"):
                data["question"] = data["query"]
        return data


class SourceItem(BaseModel):
    """Structured citation metadata delivered with generation responses."""

    id: str  # S1, S2, etc.
    product_id: str | None
    product_name: str | None
    document: str
    page: int | None
    chunk_id: str
    source_type: str
    ocr_confidence: float | None
    cited: bool = False
    suspicious: bool = False
    snippet: str
