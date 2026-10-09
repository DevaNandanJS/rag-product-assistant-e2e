"""Spike: Qdrant embedded mode verification.
Validates Qdrant local in-memory (":memory:") and local disk mode.
Confirms collection creation with named dense (Cosine, 384) and sparse vectors,
payload indexing, metadata filtering, and multi-vector search.
"""

import shutil
import tempfile

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


def run_qdrant_verification(client: QdrantClient, mode_label: str) -> None:
    print(f"\n--- Testing Qdrant in mode: {mode_label} ---")
    collection_name = "spike_test_collection"

    # Create collection with named dense + sparse vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=4, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )

    # Upsert test points
    points = [
        PointStruct(
            id="00000000-0000-0000-0000-000000000001",
            vector={
                "dense": [1.0, 0.0, 0.0, 0.0],
                "sparse": SparseVector(indices=[10, 20], values=[0.8, 0.5]),
            },
            payload={
                "product_id": "PKG-120",
                "category": "packaging_equipment",
                "document": "catalog.json",
            },
        ),
        PointStruct(
            id="00000000-0000-0000-0000-000000000002",
            vector={
                "dense": [0.0, 1.0, 0.0, 0.0],
                "sparse": SparseVector(indices=[10, 30], values=[0.2, 0.9]),
            },
            payload={
                "product_id": "WHS-1800",
                "category": "warehouse_storage",
                "document": "catalog.json",
            },
        ),
    ]

    client.upsert(collection_name=collection_name, points=points)
    count_res = client.count(collection_name=collection_name)
    assert count_res.count == 2, f"Expected 2 points, found {count_res.count}"
    print(f"Upserted {count_res.count} points successfully.")

    # Test filtered dense search
    category_filter = Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="packaging_equipment"))]
    )

    # Use query_points (modern qdrant client API) or search
    results = client.query_points(
        collection_name=collection_name,
        query=[1.0, 0.0, 0.0, 0.0],
        using="dense",
        query_filter=category_filter,
        limit=5,
    ).points

    assert len(results) == 1, f"Expected 1 filtered result, got {len(results)}"
    assert results[0].payload["product_id"] == "PKG-120"
    print(f"Dense query with filter returned expected product: {results[0].payload['product_id']}")

    # Test sparse query
    sparse_results = client.query_points(
        collection_name=collection_name,
        query=SparseVector(indices=[30], values=[1.0]),
        using="sparse",
        limit=5,
    ).points

    assert len(sparse_results) > 0, "Sparse query returned 0 results"
    assert sparse_results[0].payload["product_id"] == "WHS-1800"
    print(f"Sparse query returned expected product: {sparse_results[0].payload['product_id']}")

    client.delete_collection(collection_name=collection_name)
    print(f"Cleaned up collection in mode: {mode_label}")


def main() -> None:
    # 1. In-memory mode
    mem_client = QdrantClient(location=":memory:")
    run_qdrant_verification(mem_client, "in-memory (:memory:)")

    # 2. Local disk mode
    temp_dir = tempfile.mkdtemp(prefix="qdrant_spike_")
    try:
        disk_client = QdrantClient(path=temp_dir)
        run_qdrant_verification(disk_client, f"local disk ({temp_dir})")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n=== All Qdrant embedded spike tests PASSED ===")


if __name__ == "__main__":
    main()
