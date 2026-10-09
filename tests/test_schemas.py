"""Unit tests for canonical domain models and chunk storage contracts."""

from app.core.schemas import Chunk, Filters, ProductRecord, SpecValue, SupplierRecord


def test_product_record_creation() -> None:
    prod = ProductRecord(
        product_id="PKG-120",
        product_name="CartonPro 1200 Semi-Automatic Carton Sealer",
        category="packaging_equipment",
        supplier_id="SUP-PKG-11",
        supplier_name="PackRight Systems Pvt. Ltd.",
        country="IN",
        description="Semi-automatic carton sealer.",
        specs={
            "throughput": SpecValue(value=18, unit="cartons/min", raw="Up to 18 cartons/min"),
            "power": SpecValue(value=0.38, unit="kW", raw="0.38 kW, 230 V AC"),
            "tape_width": SpecValue(value="48-72", unit="mm", raw="48-72 mm"),
            "missing_field": SpecValue(value=None, raw=None),
        },
        source_document="catalog.json",
    )
    assert prod.product_id == "PKG-120"
    assert prod.specs["throughput"].value == 18
    assert prod.specs["missing_field"].value is None


def test_supplier_record() -> None:
    sup = SupplierRecord(
        supplier_id="SUP-PKG-11",
        supplier_name="PackRight Systems Pvt. Ltd.",
        headquarters="Pune, Maharashtra, India",
        country="IN",
        categories=["packaging_equipment"],
        regions_served=["India", "Middle East"],
        source_document="suppliers.json",
    )
    assert sup.supplier_id == "SUP-PKG-11"
    assert "India" in sup.regions_served


def test_chunk_contract() -> None:
    chunk = Chunk(
        chunk_id="PKG-120_CARD",
        point_id="12345678-1234-5678-1234-567812345678",
        text="CartonPro 1200 | Packaging | Overview",
        display_text="Overview of CartonPro 1200",
        chunk_type="card",
        product_id="PKG-120",
        product_name="CartonPro 1200",
        category="packaging_equipment",
        document="catalog.json",
        source_type="structured",
        token_count=120,
        suspicious=False,
    )
    assert chunk.chunk_id == "PKG-120_CARD"
    assert chunk.chunk_type == "card"
    assert chunk.source_type == "structured"
    assert chunk.token_count == 120


def test_filters_model() -> None:
    f = Filters(category="packaging_equipment", product_id="PKG-120")
    assert f.category == "packaging_equipment"
    assert f.country is None
