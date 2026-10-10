"""Unit tests for entity resolution, comparison query detection, and boosting."""

from __future__ import annotations

import uuid

from app.core.schemas import Chunk
from app.retrieval.entities import (
    apply_entity_boost,
    ensure_card_coverage,
    extract_entities,
    is_comparison_query,
)


def _make_chunk(cid: str, pid: str | None = None, chunk_type: str = "card") -> Chunk:
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"1|doc.json|{cid}"))
    return Chunk(
        chunk_id=cid,
        point_id=point_id,
        text=f"Body of {cid}",
        display_text=f"Body of {cid}",
        chunk_type=chunk_type,
        document="doc.json",
        product_id=pid,
        source_type="structured",
        token_count=10,
    )


class TestEntityResolution:
    def test_extract_exact_sku(self) -> None:
        ents = extract_entities("What is the throughput of the PKG-120 sealer?")
        assert ents == ["PKG-120"]

    def test_extract_sku_without_hyphen(self) -> None:
        ents = extract_entities("Specifications for pkg120 and whs1800")
        assert "PKG-120" in ents
        assert "WHS-1800" in ents

    def test_extract_alias_phrases(self) -> None:
        ents = extract_entities("Tell me about the carton sealer and turntable pallet wrapper")
        assert "PKG-120" in ents
        assert "PKG-220" in ents

    def test_distinguishes_variant_from_base(self) -> None:
        ents = extract_entities("Differences between PKG-120-PRO and PKG-120")
        assert "PKG-120-PRO" in ents
        assert "PKG-120" in ents

    def test_comparison_detection(self) -> None:
        assert is_comparison_query("Compare PKG-120 and PKG-220", ["PKG-120", "PKG-220"])
        assert is_comparison_query("PKG-120 vs PKG-220", ["PKG-120", "PKG-220"])
        assert is_comparison_query("How does PKG-120 differ from other models?", ["PKG-120"])
        assert not is_comparison_query("What is the motor wattage of TEX-12?", ["TEX-12"])

    def test_apply_entity_boost(self) -> None:
        c1 = _make_chunk("C1", pid="PKG-120")
        c2 = _make_chunk("C2", pid="REF-320")

        # Initial scores: C2 is higher
        candidates = [(c1, 1.0), (c2, 1.2)]
        # Boost PKG-120 by 1.5x -> C1 becomes 1.5, surpassing C2 (1.2)
        boosted = apply_entity_boost(candidates, entities=["PKG-120"], boost=1.5)

        assert boosted[0][0].chunk_id == "C1"
        assert boosted[0][1] == 1.5
        assert boosted[1][0].chunk_id == "C2"
        assert boosted[1][1] == 1.2

    def test_ensure_card_coverage(self) -> None:
        c1_card = _make_chunk("C1_CARD", pid="PKG-120", chunk_type="card")
        c2_spec = _make_chunk("C2_SPEC", pid="PKG-220", chunk_type="spec")
        c2_card = _make_chunk("C2_CARD", pid="PKG-220", chunk_type="card")

        candidates = [(c1_card, 0.9), (c2_spec, 0.8)]
        pool = [c1_card, c2_spec, c2_card]

        # For comparison query with PKG-120 and PKG-220, PKG-220 lacks a card chunk
        covered = ensure_card_coverage(
            candidates,
            entities=["PKG-120", "PKG-220"],
            all_chunks_pool=pool,
        )

        # C2_CARD should be injected
        chunk_ids = [c.chunk_id for c, _ in covered]
        assert "C2_CARD" in chunk_ids
