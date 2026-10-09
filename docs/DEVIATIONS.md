# Architectural Deviations Log

This document tracks all intentional divergences or external library/API adjustments from the initial architecture specified in `docs/BUILD_GUIDE.md`.

| Date | Phase | Component | Original Specification | Implemented Fallback / Deviation | Rationale & Safety Impact |
|---|---|---|---|---|---|
| 2026-10-09 | Phase 0 | LLM Models | Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) | Retained as default in config; graceful keyless fallback supported | Ensures offline reviewer compliance (R3, R4) when no external cloud keys are supplied. |
| 2026-10-09 | Phase 1 | Python Compatibility | Python 3.11+ minimum target | Accommodated host Python 3.10.11 runtime (`target-version="py310"`, `timezone.utc`) | Prevents `datetime.UTC` AttributeError on Python 3.10 while preserving full type safety. |
| 2026-10-09 | Phase 0 | FastEmbed Reranker | `from fastembed import TextCrossEncoder` | `from fastembed.rerank.cross_encoder import TextCrossEncoder` | In FastEmbed >= 0.4.0 (0.9.0), cross-encoder models were modularized under `fastembed.rerank`. Fallback try/except added. |
