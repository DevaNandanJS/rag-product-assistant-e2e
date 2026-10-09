"""Structured chunker for catalog products and suppliers.
Generates Master Card Chunks and Atomic Specification Chunks with strict token ceilings.
"""

from __future__ import annotations

import uuid
from typing import Sequence

from app.core.schemas import Chunk, ProductRecord, SpecValue, SupplierRecord
from app.ingestion.chunkers.tokens import TokenCounter

# Standard canonical comparison keys checked across products
CANONICAL_SPEC_KEYS = (
    "throughput",
    "power",
    "voltage",
    "dimensions",
    "mass",
    "capacity",
    "operating_temperature",
    "price",
    "lead_time",
    "warranty",
)


def _format_spec_value(spec: SpecValue | None) -> str:
    """Format a SpecValue into a clean string representation."""
    if spec is None or spec.value is None:
        return "Not documented"
    if spec.raw:
        return str(spec.raw).strip()
    if spec.unit:
        return f"{spec.value} {spec.unit}".strip()
    return str(spec.value).strip()


def chunk_structured_products(
    products: Sequence[ProductRecord],
    document: str,
    counter: TokenCounter,
    max_tokens: int = 480,
) -> list[Chunk]:
    """Generate Master Card Chunks and Atomic Spec Chunks for products.

    Args:
        products: List of canonical ProductRecord instances.
        document: Originating filename (e.g. 'catalog.json').
        counter: TokenCounter implementation.
        max_tokens: Hard ceiling for total tokens per chunk (default 480).

    Returns:
        List of Chunk objects (card and spec types).
    """
    chunks: list[Chunk] = []

    for product in products:
        p_id = product.product_id
        p_name = product.product_name
        cat = product.category
        sup_id = product.supplier_id or "UNKNOWN"
        sup_name = product.supplier_name or "Unknown Supplier"
        country = product.country or "IN"
        desc = product.description or ""

        # -------------------------------------------------------------------
        # 1. Master Card Chunk
        # -------------------------------------------------------------------
        card_id = f"{p_id}_CARD"
        card_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|{document}|{card_id}"))
        card_header = f"{p_name} | {cat} | Overview"

        spec_lines: list[str] = []
        # First include any specs defined on the product
        handled_keys: set[str] = set()
        for k, v in product.specs.items():
            formatted = _format_spec_value(v)
            spec_lines.append(f"  {k}: {formatted}")
            handled_keys.add(k)

        # For common comparison keys not present on product, add if not consumable/custom
        for canonical_key in CANONICAL_SPEC_KEYS:
            if canonical_key not in handled_keys and canonical_key in ("price", "lead_time", "warranty"):
                spec_lines.append(f"  {canonical_key}: Not documented")

        specs_section = "\n".join(spec_lines)
        card_body = (
            f"{p_name}\n"
            f"{desc}\n"
            f"Supplier: {sup_name} ({sup_id}), {country}\n"
            f"Specifications:\n{specs_section}"
        ).strip()

        card_full_text = f"{card_header}\n{card_body}"
        tok_count = counter.count(card_full_text)

        # Enforce max_tokens ceiling if text exceeds
        if tok_count > max_tokens:
            words = card_body.split()
            while words and counter.count(f"{card_header}\n{' '.join(words)}") > max_tokens:
                words.pop()
            card_body = " ".join(words)
            card_full_text = f"{card_header}\n{card_body}"
            tok_count = counter.count(card_full_text)

        chunks.append(
            Chunk(
                chunk_id=card_id,
                point_id=card_point_id,
                text=card_full_text,
                display_text=card_body,
                chunk_type="card",
                product_id=p_id,
                product_name=p_name,
                category=cat,
                supplier_id=product.supplier_id,
                supplier_name=product.supplier_name,
                country=country,
                document=document,
                source_type="structured",
                token_count=tok_count,
            )
        )

        # -------------------------------------------------------------------
        # 2. Atomic Specification Chunks
        # -------------------------------------------------------------------
        for spec_idx, (spec_key, spec_val) in enumerate(product.specs.items(), start=1):
            spec_chunk_id = f"{p_id}_SPEC_{spec_idx:02d}"
            spec_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|{document}|{spec_chunk_id}"))

            spec_header = f"{p_name} | {cat} | {spec_key}"
            val_formatted = _format_spec_value(spec_val)
            spec_body = f"{spec_key}: {val_formatted}"
            spec_full_text = f"{spec_header}\n{spec_body}"
            spec_tok_count = counter.count(spec_full_text)

            if spec_tok_count > max_tokens:
                words = spec_body.split()
                while words and counter.count(f"{spec_header}\n{' '.join(words)}") > max_tokens:
                    words.pop()
                spec_body = " ".join(words)
                spec_full_text = f"{spec_header}\n{spec_body}"
                spec_tok_count = counter.count(spec_full_text)

            chunks.append(
                Chunk(
                    chunk_id=spec_chunk_id,
                    point_id=spec_point_id,
                    text=spec_full_text,
                    display_text=spec_body,
                    chunk_type="spec",
                    product_id=p_id,
                    product_name=p_name,
                    category=cat,
                    supplier_id=product.supplier_id,
                    supplier_name=product.supplier_name,
                    country=country,
                    document=document,
                    section=spec_key,
                    source_type="structured",
                    token_count=spec_tok_count,
                )
            )

    return chunks


def chunk_structured_suppliers(
    suppliers: Sequence[SupplierRecord],
    document: str,
    counter: TokenCounter,
    max_tokens: int = 480,
) -> list[Chunk]:
    """Generate Overview Card Chunks for suppliers.

    Args:
        suppliers: List of canonical SupplierRecord instances.
        document: Originating filename (e.g. 'suppliers.json').
        counter: TokenCounter implementation.
        max_tokens: Hard ceiling for total tokens per chunk (default 480).

    Returns:
        List of Chunk objects (card type).
    """
    chunks: list[Chunk] = []

    for supplier in suppliers:
        s_id = supplier.supplier_id
        s_name = supplier.supplier_name
        country = supplier.country or "IN"
        card_id = f"{s_id}_CARD"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|{document}|{card_id}"))

        header = f"{s_name} | Supplier Overview"
        cats_str = ", ".join(supplier.categories) if supplier.categories else "Not documented"
        regions_str = ", ".join(supplier.regions_served) if supplier.regions_served else "Not documented"
        lead_time = supplier.lead_time_notes or "Not documented"

        body = (
            f"Supplier Name: {s_name} ({s_id})\n"
            f"Headquarters: {supplier.headquarters}\n"
            f"Country: {country}\n"
            f"Categories: {cats_str}\n"
            f"Regions Served: {regions_str}\n"
            f"Lead Time Notes: {lead_time}"
        ).strip()

        full_text = f"{header}\n{body}"
        tok_count = counter.count(full_text)

        if tok_count > max_tokens:
            words = body.split()
            while words and counter.count(f"{header}\n{' '.join(words)}") > max_tokens:
                words.pop()
            body = " ".join(words)
            full_text = f"{header}\n{body}"
            tok_count = counter.count(full_text)

        chunks.append(
            Chunk(
                chunk_id=card_id,
                point_id=point_id,
                text=full_text,
                display_text=body,
                chunk_type="card",
                supplier_id=s_id,
                supplier_name=s_name,
                country=country,
                document=document,
                source_type="structured",
                token_count=tok_count,
            )
        )

    return chunks
