# Architectural Deviations Log

This document tracks all intentional divergences or external library/API adjustments from the initial architecture specified in `docs/BUILD_GUIDE.md`.

| Date | Phase | Component | Original Specification | Implemented Fallback / Deviation | Rationale & Safety Impact |
|---|---|---|---|---|---|
| 2026-10-09 | Phase 0 | LLM Models | Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) | Retained as default in config; graceful keyless fallback supported | Ensures offline reviewer compliance (R3, R4) when no external cloud keys are supplied. |
| 2026-10-09 | Phase 1 | Python Compatibility | Python 3.11+ minimum target | Accommodated host Python 3.10.11 runtime (`target-version="py310"`, `timezone.utc`) | Prevents `datetime.UTC` AttributeError on Python 3.10 while preserving full type safety. |
| 2026-10-09 | Phase 0 | FastEmbed Reranker | `from fastembed import TextCrossEncoder` | `from fastembed.rerank.cross_encoder import TextCrossEncoder` | In FastEmbed >= 0.4.0 (0.9.0), cross-encoder models were modularized under `fastembed.rerank`. Fallback try/except added. |
| 2026-10-09 | Phase 2 | PDF Generation | `reportlab` not listed in requirements.txt | Added `reportlab==4.2.5` to requirements.txt | BUILD_GUIDE specifies reportlab for synthetic text PDF generation but omitted from pinned dependencies list. Stable library with no conflicts. |
| 2026-10-09 | Phase 3A | OCR / Tesseract | `image_to_data(output_type=Output.DATAFRAME)` | Fallback to `Output.DICT` when `pandas` is not installed | `pandas` is not in pinned requirements. `Output.DICT` produces identical coordinates and confidences with zero external dependencies. |
| 2026-10-09 | Phase 3B | Qdrant Sparse Setup | BM25 sparse vectors in Qdrant | Provisioned named sparse vector config in collection schema; vector values deferred to Phase 5 | Collection schema is prepared for hybrid search without requiring premature BM25 token generation in Phase 3B ingestion. |
| 2026-10-10 | Phase 5 | BM25 Sparse Vectors | Qdrant native IDF modifier | `SparseEmbedder` protocol with `FastEmbedSparseEmbedder` (`Qdrant/bm25`) + `FakeSparseEmbedder` | Maintains `qdrant-client==1.11.3` pin without breaking changes; pre-trained weights avoid runtime corpus-fitting state; FakeSparseEmbedder guarantees 100% offline, zero-network test execution. |
| 2026-10-10 | Phase 10 | CLI / Model Caching | Undefined `run_download_models` in `app/cli.py` | Implemented `run_download_models()` in `app/cli.py` initializing dense, sparse BM25, and reranker models into cache | Allows Docker build layer to pre-bake model weights into `/opt/models`, preventing container boot latency and runtime downloads. |
| 2026-10-10 | Phase 10 | Pytest Configuration | `asyncio_mode` missing from `pyproject.toml` | Added `asyncio_mode = "auto"` under `[tool.pytest.ini_options]` in `pyproject.toml` | Ensures automated collection and execution of asynchronous test functions across all test modules without decorating each test. |
