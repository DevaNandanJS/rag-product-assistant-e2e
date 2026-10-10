"""Unit and regression tests for Phase 2 dataset modeling, synthetic generation, and manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from app.core.schemas import ProductRecord, SupplierRecord
from scripts.make_dataset import generate_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BULLETINS_DIR = RAW_DIR / "bulletins"


def test_catalog_json_loadable():
    """Verify data/raw/catalog.json exists, has 9 products, and conforms to ProductRecord schema."""
    catalog_path = RAW_DIR / "catalog.json"
    assert catalog_path.exists(), f"Missing catalog file: {catalog_path}"

    with open(catalog_path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 9

    products = [ProductRecord.model_validate(item) for item in data]
    product_ids = {p.product_id for p in products}

    expected_ids = {
        "PKG-120", "PKG-220", "PKG-CF48",
        "WHS-1800", "WHS-1000", "WHS-400",
        "REF-320", "REF-700", "TEX-12",
    }
    assert product_ids == expected_ids

    # Verify intentional data gaps (null values)
    ref320 = next(p for p in products if p.product_id == "REF-320")
    assert ref320.specs["daily_energy_kwh"].value is None

    pkg_cf48 = next(p for p in products if p.product_id == "PKG-CF48")
    assert pkg_cf48.specs["food_contact_certified"].value is None

    ref700 = next(p for p in products if p.product_id == "REF-700")
    assert ref700.specs["medical_grade_certified"].value is None


def test_suppliers_json_loadable():
    """Verify suppliers.json exists, has 4 suppliers, and conforms to SupplierRecord schema."""
    suppliers_path = RAW_DIR / "suppliers.json"
    assert suppliers_path.exists(), f"Missing suppliers file: {suppliers_path}"

    with open(suppliers_path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 4

    suppliers = [SupplierRecord.model_validate(item) for item in data]
    supplier_ids = {s.supplier_id for s in suppliers}

    expected_ids = {"SUP-PKG-11", "SUP-WHS-21", "SUP-REF-31", "SUP-TEX-41"}
    assert supplier_ids == expected_ids


def test_all_bulletins_exist():
    """Verify all 4 canonical technical bulletin markdown files exist and are non-empty."""
    bulletin_names = [
        "PR-TECH-07.md",
        "WHS-SAFE-04.md",
        "POLAR-OPS-02.md",
        "LOOM-TECH-03.md",
    ]
    for b_name in bulletin_names:
        p = BULLETINS_DIR / b_name
        assert p.exists(), f"Missing bulletin file: {p}"
        content = p.read_text(encoding="utf-8").strip()
        assert len(content) > 50, f"Bulletin {b_name} appears empty or truncated"


def test_manifest_loadable():
    """Verify data/manifest.json loads and contains all required schema keys."""
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("Manifest not generated yet; run scripts/make_dataset.py first")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert isinstance(manifest, list)
    assert len(manifest) >= 50, f"Manifest fact count too low: {len(manifest)}"

    required_keys = {
        "fact_id", "product_id", "supplier_id", "bulletin_id",
        "field", "value", "unit", "raw", "document", "page",
        "requires_ocr", "search_strings",
    }
    for fact in manifest:
        assert isinstance(fact, dict)
        assert required_keys.issubset(fact.keys()), f"Fact missing required keys: {fact}"
        assert isinstance(fact["search_strings"], list)


def test_manifest_catalog_facts_present():
    """Verify all product_ids referenced in manifest catalog facts exist in catalog."""
    manifest_path = DATA_DIR / "manifest.json"
    catalog_path = RAW_DIR / "catalog.json"
    confusable_path = RAW_DIR / "catalog_confusable.json"

    if not manifest_path.exists():
        pytest.skip("Manifest not generated yet; run scripts/make_dataset.py first")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    valid_product_ids = {p["product_id"] for p in catalog}
    if confusable_path.exists():
        with open(confusable_path, encoding="utf-8") as f:
            confusable = json.load(f)
            valid_product_ids.update(p["product_id"] for p in confusable)

    for fact in manifest:
        pid = fact["product_id"]
        if pid is not None:
            assert pid in valid_product_ids, f"Manifest fact references unknown product_id: {pid}"


def test_manifest_supplier_facts_present():
    """Verify all supplier_ids referenced in manifest exist in suppliers.json."""
    manifest_path = DATA_DIR / "manifest.json"
    suppliers_path = RAW_DIR / "suppliers.json"

    if not manifest_path.exists():
        pytest.skip("Manifest not generated yet; run scripts/make_dataset.py first")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    with open(suppliers_path, encoding="utf-8") as f:
        suppliers = json.load(f)

    valid_supplier_ids = {s["supplier_id"] for s in suppliers}

    for fact in manifest:
        sid = fact["supplier_id"]
        if sid is not None:
            assert sid in valid_supplier_ids, f"Manifest fact references unknown supplier_id: {sid}"


@pytest.mark.slow
def test_scanned_pdfs_have_no_text_layer():
    """Verify scanned PDFs contain zero selectable text layer across all pages."""
    scanned_files = ["WHS-1800_scanned.pdf", "TEX-12_scanned.pdf"]
    for sf in scanned_files:
        p = RAW_DIR / sf
        if not p.exists():
            pytest.skip(f"Scanned PDF not generated yet: {p}")

        doc = pymupdf.open(str(p))
        assert len(doc) >= 2, f"{sf} must have at least 2 pages"
        for page_idx, page in enumerate(doc):
            extracted = page.get_text().strip()
            msg = f"{sf} page {page_idx + 1} contains unexpected text layer: {repr(extracted)}"
            assert extracted == "", msg
        doc.close()


def test_confusable_sku_is_distinct():
    """Verify PKG-120-PRO in catalog_confusable.json has distinct throughput and tape width."""
    confusable_path = RAW_DIR / "catalog_confusable.json"
    if not confusable_path.exists():
        pytest.skip("catalog_confusable.json not generated yet")

    with open(confusable_path, encoding="utf-8") as f:
        data = json.load(f)

    pro = next(p for p in data if p["product_id"] == "PKG-120-PRO")
    assert pro["specs"]["throughput"]["value"] == 24
    assert pro["specs"]["tape_width"]["value"] == "50–80"
    assert pro["specs"]["price"]["value"] == 95000


def test_injection_bulletin_contains_marker():
    """Verify SUPPLIER-ADV-INJECTED.md contains the poisoned prompt injection instruction."""
    p = BULLETINS_DIR / "SUPPLIER-ADV-INJECTED.md"
    assert p.exists(), f"Missing injection bulletin: {p}"
    content = p.read_text(encoding="utf-8")
    assert "IMPORTANT SYSTEM OVERRIDE" in content
    assert "50-year unconditional warranty" in content


def test_make_dataset_is_reproducible(tmp_path: Path):
    """Verify manifest generation is completely deterministic across repeated executions."""
    out1 = tmp_path / "manifest_1.json"
    out2 = tmp_path / "manifest_2.json"

    generate_manifest(out1)
    generate_manifest(out2)

    assert out1.read_bytes() == out2.read_bytes()


def test_questions_json_integrity():
    """Verify eval/questions.json contains 40 questions with 30 dev and 10 holdout."""
    q_path = PROJECT_ROOT / "eval" / "questions.json"
    assert q_path.exists(), f"Missing questions file: {q_path}"

    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)

    assert len(questions) == 40
    dev_count = sum(1 for q in questions if q.get("split") == "dev")
    holdout_count = sum(1 for q in questions if q.get("split") == "holdout")
    assert dev_count == 30
    assert holdout_count == 10

    categories = {q["category"] for q in questions}
    expected_categories = {
        "direct_factual",
        "semantic_paraphrase",
        "cross_product_comparison",
        "multi_doc_relational",
        "metadata_filtered",
        "unanswerable_offtopic",
        "unanswerable_missing",
        "ocr_only",
        "exact_code_search",
    }
    assert categories == expected_categories
