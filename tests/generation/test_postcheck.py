"""Offline unit tests for deterministic claim and citation post-checking."""

from app.generation.postcheck import check_citations, extract_numeric_claims, verify


def test_detects_fabricated_warranty():
    context = (
        "CartonPro 1200 Semi-Automatic Carton Sealer. Throughput: 18 cartons/min. "
        "Power: 0.38 kW, 230 V AC. Warranty: 24 months against defects."
    )
    generated = (
        "The CartonPro 1200 operates at 18 cartons/min [S1] and includes a 30 years warranty [S1]."
    )
    warnings = verify(generated, context, source_ids={"S1"})
    assert len(warnings) == 1
    assert "30 years" in warnings[0].claim
    assert "not supported" in warnings[0].reason


def test_no_warnings_when_claims_present():
    context = (
        "CartonPro 1200 specs: Throughput: 18 cartons/min. Power: 0.38 kW, 230 V AC. "
        "Warranty: 6 months."
    )
    generated = (
        "The machine operates at 18 cartons/min [S1], consumes 0.38 kW at 230 V AC [S1], "
        "and is covered by a 6 months warranty [S1]."
    )
    warnings = verify(generated, context, source_ids={"S1"})
    assert len(warnings) == 0


def test_detects_invalid_citation():
    context = "CartonPro 1200 power is 0.38 kW."
    generated = "CartonPro 1200 consumes 0.38 kW [S5]."
    warnings = verify(generated, context, source_ids={"S1", "S2"})
    assert any("[S5]" in w.claim for w in warnings)
    assert any("does not match any retrieved source ID" in w.reason for w in warnings)


def test_valid_citations_pass():
    generated = "Fact A [S1] and Fact B [S2]."
    warnings = check_citations(generated, source_ids={"S1", "S2"})
    assert len(warnings) == 0


def test_price_extraction():
    context = "Indicative catalog price: USD 15,000 with MOQ 1 unit."
    # Grounded price
    grounded = "The price is USD 15,000 [S1]."
    claims = extract_numeric_claims(grounded)
    assert "USD 15,000" in claims
    assert len(verify(grounded, context, source_ids={"S1"})) == 0

    # Fabricated price
    fabricated = "The discounted price is USD 25,000 [S1]."
    warnings = verify(fabricated, context, source_ids={"S1"})
    assert len(warnings) == 1
    assert "USD 25,000" in warnings[0].claim
