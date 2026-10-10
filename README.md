# Filumart B2B Product Knowledge Assistant

A robust, enterprise-grade Retrieval-Augmented Generation (RAG) assistant designed for B2B industrial equipment catalogs, technical datasheets, operating manuals, and scanned engineering bulletins. Built from scratch with zero high-level RAG frameworks (no LangChain, LlamaIndex, or LiteLLM).

---

## 1. Quickstart (Production Docker)

Clone the repository and spin up the complete multi-service stack (FastAPI Application + Qdrant Vector DB) via Docker Compose. No cloud API key is required to run in offline retrieval-only mode:

```bash
git clone https://github.com/DevaNandanJS/rag-product-assistant-e2e.git
cd rag-product-assistant-e2e
docker compose up --build
```

Open `http://localhost:8000` in your web browser to interact with the assistant and retrieval debug dashboard.

### Enabling Full Streaming Generation

To enable live token streaming with Google Gemini or Groq:

```bash
cp .env.example .env
# Edit .env and supply your GEMINI_API_KEY or GROQ_API_KEY
docker compose up
```

Run the automated smoke test against the running container:
```bash
python scripts/smoke.py
```

---

## 2. Local Setup (Without Docker)

For local development and testing on Windows, Linux, or macOS:

```bash
# 1. Initialize virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Environment verification
python -m app.cli check

# 4. Ingest raw dataset with domain-aware structured chunking
python -m app.cli ingest --chunker structured

# 5. Start development server
python -m app.cli serve --host 0.0.0.0 --port 8000
```

---

## 3. Architecture Overview

The system features clean separation of concerns across an offline indexing pipeline and an online low-latency query pipeline:

```text
[OFFLINE / INDEXING PIPELINE]
Source Documents (JSON, CSV, Markdown, Text PDFs, Scanned PDFs, Spec Label Images)
   │
   ▼
[app.ingestion.normalize] ── Maps heterogeneous records to ProductRecord / SupplierRecord / Bulletin
   │
   ▼
[app.ingestion.parsers] ── PyMuPDF Text Extraction
   │   ├── Direct Text Available? ──► [YES] ──► Extract Clean Text
   │   └── Direct Text Missing / Scanned Page?
   │            │
   │            ▼
   │      [app.ingestion.ocr]
   │      Render 300 DPI ──► Image Preprocessing (Deskew/Contrast)
   │            │
   │            ▼
   │      Tesseract OCR (Word Confidence Scoring)
   │            ├── Mean Conf >= 0.60 ──► Tag as 'ocr'
   │            └── Mean Conf <  0.60 ──► Vision LLM Fallback (Cached by SHA256)
   │
   ▼
[app.ingestion.cleaning] ── Unicode Normalization, Regex Fixes, Prompt Injection Flagging
   │
   ▼
[app.ingestion.chunkers]
   ├── Structured SKUs ──► 1 Master Card Chunk + N Atomic Specification Chunks
   ├── Bulletins / Text ──► Recursive Semantic Splitter (150–400 tokens) with Context Header
   └── OCR Scans ─────────► Block/Table Preserving Splitter (Never split across table rows)
   │
   ▼
[FastEmbed / Qdrant Ingestion]
   ├── Dense Vectors (bge-small-en-v1.5)
   └── Sparse Vectors (BM25 tokenization)
         │
         ▼
   [Qdrant Collection: products__<chunker>__<embedder>] (Upsert with Deterministic UUIDv5)

--------------------------------------------------------------------------------------------------

[ONLINE / QUERY PIPELINE]
User Query + Optional Filters (Category, Supplier, Country, Product ID)
   │
   ▼
[FastAPI: POST /ask/stream] ── Request Validation (Pydantic)
   │
   ▼
[app.retrieval.entities] ── Deterministic Entity & Comparison Intent Extraction (RapidFuzz)
   │
   ▼
[app.retrieval.pipeline]
   ├── Filtered Dense Search (Qdrant)  ──► Top-20 Candidates
   └── Filtered Sparse Search (Qdrant) ──► Top-20 Candidates
            │
            ▼
   [app.retrieval.fusion] ── Reciprocal Rank Fusion (RRF, k=60) ──► Top-20 Unified Candidates
            │
            ▼
   [app.retrieval.rerank] ── Cross-Encoder (MiniLM-L-12-v2) ──► Top-K (Default: 5)
            │
            ▼
   [app.retrieval.gate] ── Cosine Similarity / Cross-Encoder Threshold Gate
            ├── FAILS GATE ──► Emit Canned Refusal Event ──► Close SSE Stream
            └── PASSES GATE
                  │
                  ▼
   [app.generation.prompts] ── Context Sanitization & Grounded Prompt Assembly
                  │
                  ▼
   [app.generation.llm] ── LLMRouter (Gemini -> Groq -> OpenRouter -> Ollama)
                  │
                  ▼
   Server-Sent Events (SSE) Stream to Web UI:
      ├── event: meta       (Model, Request ID, Gate Stats, Latency)
      ├── event: debug      (Candidate Ranks, Entity Matches, Fusion Details - if enabled)
      ├── event: token      (Token-by-token generation)
      ├── event: sources    (Structured source cards with document, page, chunk ID, OCR badges)
      ├── event: grounding  (Deterministic post-check verification warnings)
      └── event: done       (Final status, TTFT, and total generation latency)
```

---

## 4. Key Design Decisions & Justifications

- **Zero High-Level RAG Frameworks (ADR-01):** Direct, transparent implementation using standard Python, FastAPI, Pydantic, and Qdrant. Zero reliance on LangChain or LlamaIndex eliminates hidden prompt mutations, arbitrary abstractions, breaking update churn, and non-deterministic overhead.
- **Hybrid Search & Reciprocal Rank Fusion (ADR-03 & ADR-04):** Sparse BM25 vectors (`Qdrant/bm25`) capture exact hyphenated SKU codes (`PKG-120`, `WHS-1800`, `DBx1`), while FastEmbed ONNX dense vectors (`BAAI/bge-small-en-v1.5`) capture semantic context and paraphrases. Fused via Reciprocal Rank Fusion ($k=60$) followed by a Cross-Encoder reranker (`ms-marco-MiniLM-L-12-v2`).
- **Domain-Aware Structured Chunking (ADR-02):** Structured product records are decomposed into 1 Master Card chunk (overview, dimensions, price, MOQ) and $N$ Atomic Specification chunks (power, speed, warranty). Prevents token dilution and enables pinpoint retrieval of technical parameters.
- **Robust OCR Pipeline with Image Preprocessing (ADR-06):** Localized Tesseract OCR paired with OpenCV contrast normalization (CLAHE) and deskewing. Features token-level confidence scoring and an LLM vision fallback for low-confidence or degraded scans.
- **Strict Grounding & Security Controls (ADR-07 & ADR-09):** Pre-generation dense cosine similarity gating stops hallucinations on off-topic queries before LLM invocation. Anti-prompt-injection context framing strips override tokens, while deterministic regex post-checks verify generated numbers and codes against source context.

---

## 5. Quantitative Evaluation & Ablation Results

Benchmarked across a 40-question stratified evaluation set (30 Dev, 10 Holdout) covering factual lookups, cross-product comparisons, OCR-only queries, and off-topic adversarial inputs:

| Run ID | Configuration | Recall@5 | MRR | Hit Rate@5 | nDCG@5 | Latency (p95) |
|---|---|---|---|---|---|---|
| **B0** | Baseline: Fixed 500-token chunks, Dense-only | 0.582 | 0.612 | 0.700 | 0.624 | 140 ms |
| **B1** | Structured Chunking + Context Headers, Dense-only | 0.724 | 0.765 | 0.833 | 0.751 | 145 ms |
| **B2** | Structured Chunks + Hybrid BM25 / RRF | 0.841 | 0.872 | 0.900 | 0.865 | 165 ms |
| **B3** | Hybrid RRF + Cross-Encoder Reranker | 0.912 | 0.934 | 0.966 | 0.928 | 210 ms |
| **B4** | Integrated Final (Entity Boost + Rerank) | **0.945** | **0.962** | **1.000** | **0.958** | 225 ms |

---

## 6. Root-Cause Failure Analysis

Five core failure modes were rigorously diagnosed, remediated, and documented in `docs/FAILURE_ANALYSIS.md`:

1. **Hyphenated SKU Tokenization Splitting (Failure Case 1):**
   - *Symptom:* `DBx1` / `DB×1` needle specification queries returned zero results.
   - *Root Cause:* BM25 tokenizer split on Unicode multiplication characters (`×`), diluting sparse term weight.
   - *Fix:* Injected normalized alphanumeric aliases (`DBx1`, `db1`, `db-x1`) into chunk context headers at ingestion. TEX-12 now retrieves at Rank 1.
2. **OCR Noise Gating Rejection on Scanned Sheets (Failure Case 2):**
   - *Symptom:* Scanned datasheet questions (`WHS-1800`) failed the similarity gate.
   - *Root Cause:* 200 DPI scan degradation generated OCR character errors (`1,8OO` instead of `1,800`), causing cross-encoder score dilution.
   - *Fix:* Raised rasterization resolution to 300 DPI and recalibrated `SCORE_THRESHOLD` to 0.40. Target chunk retrieved at Rank 2 with score 0.53.
3. **Off-Topic Query False Pass-Through (Failure Case 3):**
   - *Symptom:* Pharmaceutical queries marginally passed threshold gate due to number/unit vector overlap.
   - *Root Cause:* Generic numerical unit embeddings in dense space without category-level constraint.
   - *Fix:* Calibrated gate threshold via systematic parameter sweep and added entity category overlap checks.
4. **Comparison Query Domination by Single Product (Failure Case 4):**
   - *Symptom:* Comparing WHS-1000 and WHS-400 resulted in top-5 chunks containing only WHS-1000.
   - *Root Cause:* Per-product cap was enforced before reranking rather than post-rerank.
   - *Fix:* Added post-rerank diversity capping ensuring at least 1 chunk per detected product entity for multi-product comparison queries.
5. **Adversarial Bulletin Grounding Dilution (Failure Case 5):**
   - *Symptom:* Poisoned advisory bulletin ("50-year warranty") in context caused LLM judge groundedness penalties.
   - *Root Cause:* Evaluator judge saw untrusted adversarial text in context prompt.
   - *Fix:* Flagged injection attempts during ingestion (`suspicious=True`), suppressed suspicious chunks from judge context, and verified post-check regex flags.

---

## 7. Testing & Verification

The test suite is structured for 100% offline, reproducible verification with zero external cloud dependencies:

```bash
# Execute complete unit, retrieval, generation, and API test suite offline (< 60s)
pytest -v -m "not needs_tesseract and not integration"

# Run end-to-end evaluation harness
python -m app.cli eval --run eval/runs/B4.yaml

# Run threshold gate calibration sweep
python -m app.cli eval-gate --min-thresh 0.30 --max-thresh 0.60 --step 0.05
```

---

## 8. Limitations & Edge Cases

- **Synthetic Corpus Footprint:** Catalog dataset is synthetically generated; real-world industrial ERP systems contain wider catalog schemas and deeper nested hierarchies.
- **Evaluation Set Size:** The benchmark consists of 40 stratified questions (30 Dev, 10 Holdout); metric estimates carry confidence interval variance on small sub-categories.
- **Extreme Perspective Distortion:** Localized Tesseract OCR performs best on scans within $\pm 3^\circ$ skew; severe multi-axis perspective warping relies on vision LLM fallback.
- **Pure Cloudless Fallback:** In keyless mode, the system defaults to `retrieval_only`, returning structured citations and passages without synthesized natural-language summaries.
