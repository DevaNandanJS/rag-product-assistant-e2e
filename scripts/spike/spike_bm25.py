"""Spike: FastEmbed BM25 sparse tokenization verification.
Tests Qdrant/bm25 sparse tokenization on hyphenated SKU tokens (PKG-120, WHS-1800, 230V, 0.38kW).
Determines if tokenization splits hyphens or drops numbers.
"""

from fastembed import SparseTextEmbedding


def test_bm25_tokenization() -> None:
    print("--- Testing BM25 Sparse Embedding (Qdrant/bm25) ---")
    model = SparseTextEmbedding(model_name="Qdrant/bm25")

    tokens_to_test = [
        "PKG-120",
        "WHS-1800",
        "230V",
        "0.38kW",
        "CartonPro 1200 Semi-Automatic Carton Sealer PKG-120 48-72 mm tape",
    ]

    print("Generating sparse representations for test strings:")
    for text in tokens_to_test:
        sparse_vec = list(model.embed([text]))[0]
        # sparse_vec has .indices and .values
        indices = list(sparse_vec.indices)
        values = list(sparse_vec.values)
        print(f"\nText: '{text}'")
        print(f"  Indices count: {len(indices)}")
        print(f"  Sample indices: {indices[:6]}")
        print(f"  Sample weights: {[round(float(v), 4) for v in values[:6]]}")

    # Test query matching on exact SKU
    doc = "The PKG-120 carton sealer requires 230V power."
    query_exact = "PKG-120"
    query_unhyphenated = "PKG 120"

    doc_vec = list(model.embed([doc]))[0]
    q1_vec = list(model.query_embed(query_exact))[0]
    q2_vec = list(model.query_embed(query_unhyphenated))[0]

    def dot_product(v1, v2) -> float:
        d1 = dict(zip(v1.indices, v1.values, strict=False))
        return sum(d1.get(idx, 0.0) * val for idx, val in zip(v2.indices, v2.values, strict=False))

    score_exact = dot_product(doc_vec, q1_vec)
    score_unhyphenated = dot_product(doc_vec, q2_vec)

    print(
        "\nSparse Dot Product Scores against doc: 'The PKG-120 carton sealer requires 230V power.'"
    )
    print(f"  Query 'PKG-120': score = {score_exact:.4f}")
    print(f"  Query 'PKG 120': score = {score_unhyphenated:.4f}")

    assert score_exact > 0, "Sparse BM25 failed to match exact SKU 'PKG-120'"
    print("\nBM25 sparse tokenization spike PASSED.")


if __name__ == "__main__":
    test_bm25_tokenization()
    print("=== All BM25 spike tests PASSED ===")
