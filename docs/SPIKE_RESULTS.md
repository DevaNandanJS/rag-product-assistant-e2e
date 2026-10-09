# Spike Results & Environment Tooling Report (Phase 0)

**Date:** October 2026  
**Platform:** Windows 10/11 x64, Python 3.12, Virtual Environment `.venv`

---

## 1. FastEmbed Dense & Cross-Encoder Reranker (`spike_fastembed.py`)

- **Dense Embedding Model:** `BAAI/bge-small-en-v1.5`
  - **Inference Engine:** ONNX Runtime via FastEmbed (CPU execution)
  - **Vector Dimension:** 384 dimensions
  - **Query Method:** `embed_model.query_embed(query)` and `embed_model.embed(documents)`
  - **Performance:** CPU inference $\approx 10\text{–}15\text{ ms}$ per short text chunk. Zero PyTorch dependency.
- **Cross-Encoder Model:** `Xenova/ms-marco-MiniLM-L-12-v2`
  - **Inference Engine:** FastEmbed TextCrossEncoder
  - **Output:** Raw logits (higher is more relevant).
  - **Behavior:** Successfully distinguished packaging domain text from unrelated racking/refrigeration text.

---

## 2. BM25 Sparse Tokenization (`spike_bm25.py`)

- **Sparse Model:** `Qdrant/bm25`
- **SKU Handling (`PKG-120`, `WHS-1800`, `230V`, `0.38kW`):**
  - Tokenizer extracts sub-word and alphanumeric tokens, associating sparse weights with vocabulary indices.
  - Dot product similarity between `PKG-120` query and document mentioning `PKG-120` produces positive matching score.
  - Exact hyphenated SKUs benefit from joint sparse + dense indexing (Hybrid RRF) because sparse search strongly weights exact alphanumeric codes while dense search captures semantic queries.

---

## 3. Qdrant Embedded Mode (`spike_qdrant_embedded.py`)

- **In-Memory Mode:** `QdrantClient(location=":memory:")`
  - Supports named multi-vector schemas: `dense` (Cosine, 384 dims) and `sparse` (`SparseVectorParams`).
  - Supports payload indexing and metadata filtering (`category == "packaging_equipment"`).
  - Fully functional offline without Docker or background daemon, satisfying testing requirements.
- **Local Disk Mode:** `QdrantClient(path="./qdrant_local")`
  - Successfully creates collections and persists points to a local directory.

---

## 4. LLM Streaming & Provider Compatibility (`spike_llm_stream.py`)

- **Client Library:** `openai.AsyncOpenAI` (v1.51.0+)
- **Google Gemini Endpoint:**
  - Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
  - Model: `gemini-3.1-flash-lite` (or current 3.x Flash tier)
  - Compatibility: Standard OpenAI chat completions endpoint streaming token-by-token.
- **Groq Endpoint:**
  - Base URL: `https://api.groq.com/openai/v1`
  - Model: `llama-3.3-70b-versatile`
- **Fallback / Keyless Behavior:**
  - If keys are missing or placeholder strings (`your_key_here`), system degrades to `retrieval_only` without throwing unhandled exceptions.

---

## 5. Tesseract OCR Discovery & Confidence (`spike_tesseract.py`)

- **Binary Resolution Strategy:**
  1. `TESSERACT_CMD` environment variable / setting
  2. Standard Windows installation path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
  3. `shutil.which("tesseract")` (system `PATH`)
- **Per-Word Confidence:**
  - Extracted via `pytesseract.image_to_data(img, output_type=Output.DICT)`.
  - Values filtered for `conf >= 0` and averaged to produce mean page confidence in $[0.0, 1.0]$.
  - If binary is absent on host, system issues clear diagnostic notice and falls back gracefully.

---

## 6. SSE Disconnect & Cancellation (`spike_sse_disconnect.py`)

- **Async Streaming Behavior:**
  - When the client disconnects or aborts the HTTP connection, `asyncio.CancelledError` is raised in the generator.
  - The `finally` block executes cleanly to release open streams and terminate any upstream LLM calls.
