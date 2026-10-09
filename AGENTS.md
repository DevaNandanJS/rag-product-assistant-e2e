# Agent Instructions: Filumart RAG Product Assistant

## 1. Operating Rules & Guardrails
- **Reference Document:** The authoritative architecture and specification document is located at `docs/BUILD_GUIDE.md`.
- **Phase Execution:** Work strictly **one phase at a time** (Phase 0 through Phase 12).
- **No Premature Execution:** Never begin Phase N+1 until explicitly instructed by the user.
- **Human Checkpoints:** Every phase defines a Human Checkpoint. At the end of Phase N, stop and report:
  1. What was built and verified.
  2. Any discrepancies or failures recorded in `docs/DEVIATIONS.md`.
  3. Commands run and verification output.
  4. Explicit request for human sign-off before proceeding.
- **Git Discipline:** DO not commit anything or do anything related to git unless user specifies. I will do it myself.

---

## 2. Hard "Do Not" List
- **No LLM/RAG Frameworks:** Absolutely NO LangChain, LlamaIndex, LiteLLM, RAGAS, or DeepEval. Write standard, modular Python.
- **No Deprecated Models:** Do not reference or use Gemini 2.5 models (retired). Default to current Gemini 3.x models (`gemini-3.1-flash-lite`) via config.
- **No Hardcoded Values:** Model names, endpoints, file paths, and thresholds must come from `pydantic-settings` (`app/core/config.py`).
- **No UI Frameworks:** No React, Vite, Streamlit, or Gradio. Frontend is vanilla HTML/JS with vendored `marked.js` and `DOMPurify`.
- **No Fake Streaming:** Tokens must yield directly from the provider iterator via Server-Sent Events (`POST /ask/stream`). Full-response buffering is prohibited.
- **No RRF Score Gating:** Never gate "no-result" answers using RRF scores. Use dense cosine similarity or reranker cross-entropy scores.
- **No `innerHTML`:** Never render untrusted text or chunk snippets via `innerHTML`. Use `DOMPurify.sanitize()` or DOM manipulation (`textContent`, `createElement`).
- **No Secret Leaks:** Never commit `.env` or print API keys in terminal logs.

---

## 3. Reviewer-First Standards
- **Reviewer Usability:** The project must start with zero cloud dependencies (`LLM_MODE=retrieval_only` or fallback mode) and pass `pytest` offline with fake embedders/LLMs.
- **Clean Fallbacks:** If API keys or external services (Tesseract, Qdrant) are missing, print clear, actionable startup notices—never an unhandled traceback.
- **Deterministic Artifacts:** Chunks, point IDs, and evaluations must be idempotent. Avoid non-deterministic UUIDs or run timestamps in vector payloads.

---

## 4. Phase Execution Prompt Template
When prompted to begin a phase, execute according to this format:

```text
Phase: [Phase Number - Name]
Objective: [Short description from Section 8 of docs/BUILD_GUIDE.md]
Files to touch: [List paths]
Verification command: [e.g., python -m app.cli test / pytest]