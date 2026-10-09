"""Spike: FastEmbed dense embeddings and cross-encoder reranker verification.
Validates loading BAAI/bge-small-en-v1.5 and Xenova/ms-marco-MiniLM-L-12-v2 via FastEmbed (ONNX).
"""

import time

from fastembed import TextEmbedding

try:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
except ImportError:
    from fastembed import TextCrossEncoder  # type: ignore


def test_dense_embedding() -> None:
    print("--- 1. Testing Dense Embedding (BAAI/bge-small-en-v1.5) ---")
    start = time.perf_counter()
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    init_time = time.perf_counter() - start
    print(f"Model initialized in {init_time:.2f}s")

    docs = [
        "CartonPro 1200 Semi-Automatic Carton Sealer PKG-120 tape width 48-72 mm",
        "StackStore 1800 Heavy Duty Pallet Racking WHS-1800 rated capacity 1800 kg",
    ]

    start = time.perf_counter()
    embeddings = list(embed_model.embed(docs))
    embed_time = time.perf_counter() - start

    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
    dim = len(embeddings[0])
    print(f"Generated {len(embeddings)} embeddings in {embed_time * 1000:.2f}ms (dimension: {dim})")
    assert dim == 384, f"Expected 384 dimensions for bge-small-en-v1.5, got {dim}"

    # Test query instruction requirement if any
    query = "Find packaging machine for carton sealing"
    query_emb = list(embed_model.query_embed(query))[0]
    assert len(query_emb) == 384, f"Expected query embedding dim 384, got {len(query_emb)}"
    print("Dense embedding spike PASSED.\n")


def test_cross_encoder_reranker() -> None:
    print("--- 2. Testing Cross-Encoder (Xenova/ms-marco-MiniLM-L-12-v2) ---")
    start = time.perf_counter()
    reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-12-v2")
    init_time = time.perf_counter() - start
    print(f"Reranker initialized in {init_time:.2f}s")

    query = "carton sealing machine tape width"
    documents = [
        "StackStore 1800 Heavy Duty Pallet Racking for warehouse storage.",
        "CartonPro 1200 Semi-Automatic Carton Sealer compatible with 48-72 mm tape.",
        "FrostHarbor CF320 Commercial Chest Freezer with -18 to -24 C range.",
    ]

    start = time.perf_counter()
    scores = list(reranker.rerank(query, documents))
    rerank_time = time.perf_counter() - start

    print(f"Reranked {len(documents)} documents in {rerank_time * 1000:.2f}ms")
    for doc, score in zip(documents, scores, strict=True):
        print(f"  Score: {score:+.4f} | Doc: {doc[:50]}...")

    # The packaging document should score highest
    assert scores[1] > scores[0], "Expected packaging document to rank higher than racking"
    assert scores[1] > scores[2], "Expected packaging document to rank higher than freezer"
    print("Cross-encoder reranker spike PASSED.\n")


if __name__ == "__main__":
    test_dense_embedding()
    test_cross_encoder_reranker()
    print("=== All FastEmbed spike tests PASSED ===")
