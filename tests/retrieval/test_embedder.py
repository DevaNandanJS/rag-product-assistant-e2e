"""Unit tests for Embedder protocol and FakeEmbedder implementation."""

from __future__ import annotations

import math

from app.retrieval.embedder import Embedder, FakeEmbedder


class TestFakeEmbedder:
    """Verifies FakeEmbedder properties for testing and CI."""

    def test_satisfies_protocol(self) -> None:
        embedder = FakeEmbedder()
        assert isinstance(embedder, Embedder)

    def test_dimensions(self) -> None:
        embedder = FakeEmbedder(dim=384)
        assert embedder.dim == 384
        vectors = embedder.embed(["Hello world", "Packaging machine"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        assert len(vectors[1]) == 384

    def test_deterministic(self) -> None:
        embedder = FakeEmbedder(dim=384)
        v1 = embedder.embed(["CartonPro 1200 sealer"])
        v2 = embedder.embed(["CartonPro 1200 sealer"])
        assert v1[0] == v2[0]

        v_diff = embedder.embed(["FlexWrap 220 wrapper"])
        assert v1[0] != v_diff[0]

    def test_l2_normalized(self) -> None:
        embedder = FakeEmbedder(dim=384)
        vectors = embedder.embed(["Sample document to test L2 norm."])
        norm = math.sqrt(sum(x * x for x in vectors[0]))
        assert math.isclose(norm, 1.0, rel_tol=1e-5)

    def test_empty_batch(self) -> None:
        embedder = FakeEmbedder(dim=384)
        assert embedder.embed([]) == []


class TestFakeSparseEmbedder:
    """Verifies FakeSparseEmbedder properties for testing and CI."""

    def test_satisfies_protocol(self) -> None:
        from app.retrieval.embedder import FakeSparseEmbedder, SparseEmbedder

        sparse = FakeSparseEmbedder()
        assert isinstance(sparse, SparseEmbedder)

    def test_deterministic_and_matching(self) -> None:
        from app.retrieval.embedder import FakeSparseEmbedder

        sparse = FakeSparseEmbedder()
        s1 = sparse.embed(["PKG-120 carton sealer"])[0]
        s2 = sparse.query_embed("PKG-120 carton sealer")

        assert s1.indices == s2.indices
        assert s1.values == s2.values
        assert len(s1.indices) > 0

    def test_different_tokens(self) -> None:
        from app.retrieval.embedder import FakeSparseEmbedder

        sparse = FakeSparseEmbedder()
        s1 = sparse.embed(["PKG-120"])[0]
        s2 = sparse.embed(["REF-320"])[0]

        # Different tokens have different indices
        assert s1.indices != s2.indices

