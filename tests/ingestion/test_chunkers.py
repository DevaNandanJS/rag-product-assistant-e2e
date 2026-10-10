"""Unit tests for chunking strategies: fixed, structured, and prose chunkers."""

from __future__ import annotations

from app.core.schemas import ProductRecord, SpecValue, SupplierRecord
from app.ingestion.chunkers.fixed import chunk_fixed
from app.ingestion.chunkers.prose import chunk_prose
from app.ingestion.chunkers.structured import (
    chunk_structured_products,
    chunk_structured_suppliers,
)
from app.ingestion.chunkers.tokens import WhitespaceTokenCounter


class TestFixedChunker:
    """Tests for Baseline B0 fixed-size sliding window chunker."""

    def test_basic_split(self) -> None:
        counter = WhitespaceTokenCounter()
        words = ["word"] * 600
        text = " ".join(words)
        chunks = chunk_fixed(
            text=text,
            document="manual.txt",
            page=1,
            source_type="extracted",
            ocr_confidence=None,
            counter=counter,
            window=100,
            overlap=20,
        )
        assert len(chunks) > 1
        for c in chunks:
            assert c.chunk_type == "fixed"
            assert c.document == "manual.txt"
            assert c.page == 1
            assert c.source_type == "extracted"
            assert c.token_count <= 120  # bounded by window

    def test_overlap_present(self) -> None:
        counter = WhitespaceTokenCounter()
        words = [f"tok_{i}" for i in range(250)]
        text = " ".join(words)
        chunks = chunk_fixed(
            text=text,
            document="doc.txt",
            page=None,
            source_type="extracted",
            ocr_confidence=None,
            counter=counter,
            window=50,
            overlap=15,
        )
        assert len(chunks) >= 2
        # Verify that trailing tokens of chunk 0 appear in chunk 1
        c0_tokens = set(chunks[0].text.split()[-10:])
        c1_tokens = set(chunks[1].text.split()[:15])
        assert len(c0_tokens.intersection(c1_tokens)) > 0

    def test_short_text_single_chunk(self) -> None:
        counter = WhitespaceTokenCounter()
        text = "Short text under window limit."
        chunks = chunk_fixed(
            text=text,
            document="short.txt",
            page=1,
            source_type="extracted",
            ocr_confidence=None,
            counter=counter,
            window=500,
        )
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_id == "short.txt_FIX_000"

    def test_no_context_headers(self) -> None:
        counter = WhitespaceTokenCounter()
        text = "Raw content with no metadata headers."
        chunks = chunk_fixed(
            text=text,
            document="raw.txt",
            page=None,
            source_type="extracted",
            ocr_confidence=None,
            counter=counter,
            window=500,
        )
        assert len(chunks) == 1
        assert chunks[0].text == chunks[0].display_text

    def test_deterministic_point_ids(self) -> None:
        counter = WhitespaceTokenCounter()
        text = "Consistent point ID verification text."
        chunks1 = chunk_fixed(
            text=text,
            document="a.txt",
            page=1,
            source_type="extracted",
            ocr_confidence=None,
            counter=counter,
        )
        chunks2 = chunk_fixed(
            text=text,
            document="a.txt",
            page=1,
            source_type="extracted",
            ocr_confidence=None,
            counter=counter,
        )
        assert chunks1[0].point_id == chunks2[0].point_id


class TestStructuredChunker:
    """Tests for Master Card and Atomic Spec chunks."""

    def test_card_and_spec_generation(self) -> None:
        counter = WhitespaceTokenCounter()
        product = ProductRecord(
            product_id="PKG-120",
            product_name="CartonPro 1200",
            category="packaging_equipment",
            supplier_id="SUP-PKG-11",
            supplier_name="PackRight Systems Pvt. Ltd.",
            country="IN",
            description="Semi-automatic carton sealer for uniform cartons.",
            specs={
                "throughput": SpecValue(value=18, unit="ctn/min", raw="18 ctn/min"),
                "power": SpecValue(value=0.38, unit="kW", raw="0.38 kW"),
                "voltage": SpecValue(value="230V AC", unit="V", raw="230V AC"),
            },
            source_document="catalog.json",
        )

        chunks = chunk_structured_products([product], document="catalog.json", counter=counter)
        # 1 card chunk + 3 spec chunks
        assert len(chunks) == 4

        card_chunk = chunks[0]
        assert card_chunk.chunk_type == "card"
        assert card_chunk.chunk_id == "PKG-120_CARD"
        assert "Overview" in card_chunk.text
        assert "PackRight Systems Pvt. Ltd." in card_chunk.text
        assert "throughput: 18 ctn/min" in card_chunk.text
        assert "price: Not documented" in card_chunk.text  # Missing common comparison field

        spec_chunks = chunks[1:]
        spec_ids = [s.chunk_id for s in spec_chunks]
        assert "PKG-120_SPEC_01" in spec_ids
        assert "PKG-120_SPEC_02" in spec_ids
        assert "PKG-120_SPEC_03" in spec_ids
        for sc in spec_chunks:
            assert sc.chunk_type == "spec"
            assert "CartonPro 1200 | packaging_equipment" in sc.text
            assert sc.display_text != sc.text  # display_text excludes header

    def test_missing_spec_value_renders_not_documented(self) -> None:
        counter = WhitespaceTokenCounter()
        product = ProductRecord(
            product_id="TEST-01",
            product_name="Test Product",
            category="testing",
            specs={"custom_metric": SpecValue(value=None)},
            source_document="catalog.json",
        )
        chunks = chunk_structured_products([product], document="catalog.json", counter=counter)
        spec_chunk = chunks[1]
        assert "custom_metric: Not documented" in spec_chunk.text

    def test_max_token_ceiling_enforced(self) -> None:
        counter = WhitespaceTokenCounter()
        long_desc = "description word " * 500
        product = ProductRecord(
            product_id="LONG-01",
            product_name="Long Product",
            category="testing",
            description=long_desc,
            specs={"throughput": SpecValue(value=10)},
            source_document="catalog.json",
        )
        chunks = chunk_structured_products(
            [product],
            document="catalog.json",
            counter=counter,
            max_tokens=480,
        )
        for c in chunks:
            assert c.token_count <= 480

    def test_supplier_card_chunk(self) -> None:
        counter = WhitespaceTokenCounter()
        supplier = SupplierRecord(
            supplier_id="SUP-01",
            supplier_name="Alpha Supplies",
            headquarters="Mumbai",
            country="IN",
            categories=["packaging_equipment", "warehouse_storage"],
            regions_served=["India", "Middle East"],
            lead_time_notes="Standard 10 days",
            source_document="suppliers.json",
        )
        chunks = chunk_structured_suppliers([supplier], document="suppliers.json", counter=counter)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.chunk_type == "card"
        assert c.chunk_id == "SUP-01_CARD"
        assert "Alpha Supplies | Supplier Overview" in c.text
        assert "Mumbai" in c.text
        assert "Standard 10 days" in c.text


class TestProseChunker:
    """Tests for recursive semantic prose chunker."""

    def test_basic_prose_split(self) -> None:
        counter = WhitespaceTokenCounter()
        para1 = "This is the first paragraph of technical specifications. " * 10
        para2 = "This is the second section addressing preventive maintenance. " * 10
        text = f"{para1}\n\n{para2}"

        chunks = chunk_prose(
            text=text,
            document="bulletin.md",
            product_id="PKG-120",
            product_name="CartonPro 1200",
            category="packaging_equipment",
            section="Maintenance",
            counter=counter,
            target_tokens=50,
            max_tokens=480,
        )
        assert len(chunks) >= 2
        for c in chunks:
            assert c.chunk_type == "prose"
            assert "CartonPro 1200 | packaging_equipment | Maintenance" in c.text
            assert c.token_count <= 480

    def test_table_rows_preserved(self) -> None:
        counter = WhitespaceTokenCounter()
        table = (
            "| Component | Interval | Lubricant |\n"
            "|---|---|---|\n"
            "| Main Bearings | 500 hrs | ISO VG 220 |\n"
            "| Drive Chain | 250 hrs | Heavy Oil |\n"
            "| Conveyor Rollers | 1000 hrs | Lithium Grease |"
        )
        chunks = chunk_prose(
            text=table,
            document="bulletin.md",
            product_id="PKG-120",
            counter=counter,
            target_tokens=250,
            max_tokens=480,
        )
        assert len(chunks) == 1
        # Entire table kept together
        assert "| Main Bearings | 500 hrs | ISO VG 220 |" in chunks[0].text
        assert "| Drive Chain | 250 hrs | Heavy Oil |" in chunks[0].text

    def test_strict_token_ceiling(self) -> None:
        counter = WhitespaceTokenCounter()
        long_prose = "Continuous run of technical prose text for stress testing token bounds. " * 80
        chunks = chunk_prose(
            text=long_prose,
            document="long_bulletin.txt",
            counter=counter,
            target_tokens=100,
            max_tokens=200,
        )
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 200
