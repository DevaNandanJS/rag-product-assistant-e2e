"""Unit and integration tests for document parsers (app.ingestion.parsers)."""

import json
from pathlib import Path

import pytest

from app.ingestion.parsers import (
    ParsedDocument,
    parse_csv,
    parse_json,
    parse_markdown,
    parse_pdf,
    parse_txt,
)

RAW_DATA_DIR = Path("data/raw")


class TestParseJson:
    """Tests for parse_json."""

    def test_parse_catalog_json_real_file(self) -> None:
        path = RAW_DATA_DIR / "catalog.json"
        if not path.exists():
            pytest.skip(f"{path} not found")

        doc = parse_json(path)
        assert isinstance(doc, ParsedDocument)
        assert len(doc.product_records) > 0
        assert doc.source_file == "catalog.json"

        # Check first product structure
        prod = doc.product_records[0]
        assert prod.product_id != ""
        assert prod.product_name != ""
        assert prod.category != ""
        assert isinstance(prod.specs, dict)

    def test_parse_suppliers_json_real_file(self) -> None:
        path = RAW_DATA_DIR / "suppliers.json"
        if not path.exists():
            pytest.skip(f"{path} not found")

        doc = parse_json(path)
        assert isinstance(doc, ParsedDocument)
        assert len(doc.supplier_records) > 0
        assert doc.source_file == "suppliers.json"

        sup = doc.supplier_records[0]
        assert sup.supplier_id.startswith("SUP-")
        assert sup.supplier_name != ""
        assert len(sup.categories) > 0

    def test_parse_json_single_object(self, tmp_path: Path) -> None:
        sample_path = tmp_path / "single_product.json"
        sample_path.write_text(
            json.dumps({
                "product_id": "TEST-1",
                "product_name": "Test Machine",
                "category": "testing",
                "specs": {"voltage": {"value": 230, "unit": "V"}},
            }),
            encoding="utf-8",
        )

        doc = parse_json(sample_path)
        assert len(doc.product_records) == 1
        assert doc.product_records[0].product_id == "TEST-1"
        assert doc.product_records[0].specs["voltage"].value == 230

    def test_parse_json_invalid_syntax_raises(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "broken.json"
        bad_json.write_text("{not: valid json}", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_json(bad_json)

    def test_parse_json_non_list_non_dict_raises(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "scalar.json"
        bad_json.write_text('"just a string"', encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a JSON array or object"):
            parse_json(bad_json)


class TestParseCsv:
    """Tests for parse_csv."""

    def test_parse_csv_products(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "product_id,product_name,category,country,description\n"
            "PKG-100,Mini Sealer,packaging_equipment,IN,Small carton sealer\n"
            "PKG-200,Maxi Sealer,packaging_equipment,IN,Large carton sealer\n",
            encoding="utf-8",
        )

        doc = parse_csv(csv_file)
        assert len(doc.product_records) == 2
        assert doc.product_records[0].product_id == "PKG-100"
        assert doc.product_records[0].product_name == "Mini Sealer"
        assert doc.product_records[1].product_id == "PKG-200"

    def test_parse_csv_suppliers(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "suppliers.csv"
        csv_file.write_text(
            "supplier_id,supplier_name,headquarters,country,categories,regions_served\n"
            "SUP-01,Alpha Ltd,Mumbai,IN,\"packaging, textile\",\"India, South Asia\"\n",
            encoding="utf-8",
        )

        doc = parse_csv(csv_file)
        assert len(doc.supplier_records) == 1
        sup = doc.supplier_records[0]
        assert sup.supplier_id == "SUP-01"
        assert "packaging" in sup.categories
        assert "South Asia" in sup.regions_served

    def test_parse_csv_empty(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")
        doc = parse_csv(csv_file)
        assert len(doc.product_records) == 0
        assert len(doc.supplier_records) == 0


class TestParseMarkdown:
    """Tests for parse_markdown."""

    def test_parse_markdown_bulletin_real_file(self) -> None:
        bulletin_path = RAW_DATA_DIR / "bulletins" / "LOOM-TECH-03.md"
        if not bulletin_path.exists():
            pytest.skip(f"{bulletin_path} not found")

        doc = parse_markdown(bulletin_path)
        assert len(doc.bulletin_records) == 1
        bulletin = doc.bulletin_records[0]
        assert bulletin.bulletin_id == "LOOM-TECH-03"
        assert "TEX-12" in bulletin.related_product_ids
        assert "lockstitch" in bulletin.content.lower()

    def test_parse_markdown_generic_sections(self, tmp_path: Path) -> None:
        md_file = tmp_path / "manual.md"
        md_file.write_text(
            "Introductory preamble before any heading.\n\n"
            "# Overview\n"
            "This is the general overview.\n\n"
            "## Technical Specifications\n"
            "- Voltage: 230V\n"
            "- Weight: 50kg\n\n"
            "# Safety Guidelines\n"
            "Always wear gloves.\n",
            encoding="utf-8",
        )

        doc = parse_markdown(md_file)
        assert len(doc.sections) == 4

        # Preamble
        assert doc.sections[0].heading == ""
        assert "Introductory preamble" in doc.sections[0].body
        assert doc.sections[0].level == 0

        # Section 1
        assert doc.sections[1].heading == "Overview"
        assert "general overview" in doc.sections[1].body
        assert doc.sections[1].level == 1

        # Section 2
        assert doc.sections[2].heading == "Technical Specifications"
        assert "Voltage: 230V" in doc.sections[2].body
        assert doc.sections[2].level == 2

        # Section 3
        assert doc.sections[3].heading == "Safety Guidelines"
        assert "Always wear gloves." in doc.sections[3].body
        assert doc.sections[3].level == 1


class TestParseTxt:
    """Tests for parse_txt."""

    def test_parse_txt_paragraphs(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text(
            "First paragraph of notes with details.\nLine 2 of para 1.\n\n\n"
            "Second paragraph with maintenance schedules.\n\n"
            "Third paragraph with contact info.\n",
            encoding="utf-8",
        )

        doc = parse_txt(txt_file)
        assert len(doc.sections) == 3
        assert "First paragraph" in doc.sections[0].body
        assert "Second paragraph" in doc.sections[1].body
        assert "Third paragraph" in doc.sections[2].body
        for section in doc.sections:
            assert section.heading == ""
            assert section.level == 0


class TestParsePdf:
    """Tests for parse_pdf."""

    @pytest.mark.slow
    def test_parse_pdf_native_text(self) -> None:
        pdf_path = RAW_DATA_DIR / "PKG-120_datasheet.pdf"
        if not pdf_path.exists():
            pytest.skip(f"{pdf_path} not found")

        doc = parse_pdf(pdf_path)
        assert len(doc.pages) > 0
        assert doc.source_file == "PKG-120_datasheet.pdf"

        # Native PDF should have extractable text and not need OCR
        for page in doc.pages:
            assert len(page.text.strip()) > 50
            assert page.needs_ocr is False
            assert "PKG-120" in page.text or "CartonPro" in page.text

    @pytest.mark.slow
    def test_parse_pdf_scanned_flags_ocr(self) -> None:
        pdf_path = RAW_DATA_DIR / "WHS-1800_scanned.pdf"
        if not pdf_path.exists():
            pytest.skip(f"{pdf_path} not found")

        doc = parse_pdf(pdf_path)
        assert len(doc.pages) > 0
        assert doc.source_file == "WHS-1800_scanned.pdf"

        # Scanned PDF should flag needs_ocr = True on its image pages
        assert any(page.needs_ocr is True for page in doc.pages)
        # Direct extractable text should be empty or near-empty
        assert all(len(page.text.strip()) < 50 for page in doc.pages if page.needs_ocr)
