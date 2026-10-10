"""Tests for the evidence-based labeling engine."""

from __future__ import annotations

from app.core.schemas import Chunk
from app.evaluation.labels import (
    EvidenceItem,
    covered_evidence_indices,
    is_chunk_match,
    label_retrieval,
    normalize_text,
)


def make_dummy_chunk(
    chunk_id: str,
    text: str,
    document: str = "catalog.json",
    page: int | None = None,
    display_text: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        point_id=f"pt_{chunk_id}",
        text=text,
        display_text=display_text or text,
        chunk_type="spec",
        document=document,
        page=page,
        source_type="structured",
        token_count=len(text.split()),
    )


class TestTextNormalization:
    def test_unicode_and_case(self) -> None:
        assert normalize_text("CartonPro 1200") == "cartonpro 1200"

    def test_dash_normalization(self) -> None:
        # en-dash, em-dash, minus sign all become hyphen
        assert normalize_text("48–72 mm") == "48-72 mm"
        assert normalize_text("48—72 mm") == "48-72 mm"
        assert normalize_text("48−72 mm") == "48-72 mm"

    def test_multiplication_sign(self) -> None:
        assert normalize_text("2400×1100×4000") == "2400x1100x4000"

    def test_degree_symbols(self) -> None:
        assert normalize_text("+2℃ to +8℃") == "+2°c to +8°c"
        assert normalize_text("-18º to -24º") == "-18° to -24°"

    def test_whitespace_collapsing(self) -> None:
        assert normalize_text("  hello \t  world \n ") == "hello world"
        assert normalize_text("") == ""


class TestChunkMatching:
    def test_exact_document_and_fact_match(self) -> None:
        chunk = make_dummy_chunk(
            chunk_id="c1",
            text="Throughput: 18 ctn/min for standard cartons.",
            document="catalog.json",
        )
        evidence = EvidenceItem(
            document="catalog.json",
            page=None,
            key_fact="18 ctn/min",
        )
        assert is_chunk_match(chunk, evidence) is True

    def test_document_mismatch(self) -> None:
        chunk = make_dummy_chunk(
            chunk_id="c1",
            text="Throughput: 18 ctn/min for standard cartons.",
            document="other_doc.json",
        )
        evidence = EvidenceItem(
            document="catalog.json",
            page=None,
            key_fact="18 ctn/min",
        )
        assert is_chunk_match(chunk, evidence) is False

    def test_document_basename_insensitivity(self) -> None:
        chunk = make_dummy_chunk(
            chunk_id="c1",
            text="Throughput: 18 ctn/min.",
            document="data/raw/catalog.json",
        )
        evidence = EvidenceItem(
            document="CATALOG.JSON",
            page=None,
            key_fact="18 ctn/min",
        )
        assert is_chunk_match(chunk, evidence) is True

    def test_page_matching_logic(self) -> None:
        # Match when both have identical page
        c_p1 = make_dummy_chunk("c1", "1,800 kg UDL", document="scan.pdf", page=1)
        ev_p1 = EvidenceItem(document="scan.pdf", page=1, key_fact="1,800 kg UDL")
        assert is_chunk_match(c_p1, ev_p1) is True

        # Mismatch when page differs
        ev_p2 = EvidenceItem(document="scan.pdf", page=2, key_fact="1,800 kg UDL")
        assert is_chunk_match(c_p1, ev_p2) is False

        # If evidence does not restrict page, any page matches
        ev_any = EvidenceItem(document="scan.pdf", page=None, key_fact="1,800 kg UDL")
        assert is_chunk_match(c_p1, ev_any) is True

        # If chunk does not specify page, it matches
        c_nopage = make_dummy_chunk("c2", "1,800 kg UDL", document="scan.pdf", page=None)
        assert is_chunk_match(c_nopage, ev_p1) is True

    def test_fuzzy_matching_for_ocr_text(self) -> None:
        # Slight OCR typo or noise
        chunk = make_dummy_chunk(
            chunk_id="c_ocr",
            text="WHS-1800 Capacity: 1,8OO kg UDL whole bay",  # Note letter O instead of 0
            document="WHS-1800_scanned.pdf",
        )
        evidence = EvidenceItem(
            document="WHS-1800_scanned.pdf",
            page=None,
            key_fact="1,800 kg UDL",
        )
        assert is_chunk_match(chunk, evidence, fuzzy_threshold=80.0) is True


class TestLabelRetrievalAndCoverage:
    def test_label_retrieval_binary_mask(self) -> None:
        c1 = make_dummy_chunk("c1", "Capacity: 1000 kg", document="catalog.json")
        c2 = make_dummy_chunk("c2", "Unrelated text about motors", document="catalog.json")
        c3 = make_dummy_chunk("c3", "Price: 2500 kg capacity", document="catalog.json")

        evidence = [
            EvidenceItem("catalog.json", None, "1000 kg"),
            EvidenceItem("catalog.json", None, "2500 kg"),
        ]

        labels = label_retrieval([c1, c2, c3], evidence)
        assert labels == [True, False, True]

    def test_covered_evidence_indices(self) -> None:
        c1 = make_dummy_chunk("c1", "Capacity: 1000 kg", document="catalog.json")
        c2 = make_dummy_chunk("c2", "Unrelated text", document="catalog.json")

        ev1 = EvidenceItem("catalog.json", None, "1000 kg")
        ev2 = EvidenceItem("catalog.json", None, "2500 kg")

        covered = covered_evidence_indices([c1, c2], [ev1, ev2])
        assert covered == {0}

    def test_empty_evidence_yields_all_false(self) -> None:
        c1 = make_dummy_chunk("c1", "Any text", document="catalog.json")
        assert label_retrieval([c1], []) == [False]
        assert covered_evidence_indices([c1], []) == set()
