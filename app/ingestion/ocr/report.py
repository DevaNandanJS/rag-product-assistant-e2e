"""OCR verification and accuracy reporting script.

Can be run directly via:
    python -m app.ingestion.ocr.report

Inspects OCR confidence, ground-truth fact recovery from manifest.json,
and character error rate (CER) across sample scans and spec images.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.ingestion.ocr.preprocess import preprocess
from app.ingestion.ocr.tesseract import TesseractEngine

logger = logging.getLogger("ocr_report")


def _compute_levenshtein(s1: str, s2: str) -> int:
    """Compute standard Levenshtein distance between two strings."""
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(s1, s2))
    except ImportError:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]


def _compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) between reference and hypothesis."""
    if not reference:
        return 0.0
    dist = _compute_levenshtein(reference, hypothesis)
    return float(dist) / float(len(reference))


def generate_ocr_report() -> int:
    """Run OCR verification across sample scanned documents and print formatted report."""
    settings = Settings()
    raw_dir = Path(settings.DATA_DIR)
    manifest_path = Path("data/manifest.json")

    print("\n" + "=" * 80)
    print(" FILUMART RAG ASSISTANT - OCR BENCHMARK & VERIFICATION REPORT")
    print("=" * 80)

    # 1. Check Tesseract availability
    try:
        engine = TesseractEngine(cmd_path=settings.TESSERACT_CMD)
        if not engine.is_available:
            resolved_hint = settings.TESSERACT_CMD or "C:\\Program Files\\Tesseract-OCR or PATH"
            raise RuntimeError(f"Tesseract binary not found (searched: {resolved_hint}).")
    except Exception as exc:
        print("\n[!] NOTICE: Tesseract OCR is not available on this system.")
        print(f"    Detail: {exc}")
        print("    Configure TESSERACT_CMD in your .env or install Tesseract-OCR.")
        print("    (Reviewer Mode: Gracefully skipping live OCR scan report)\n")
        return 0

    # 2. Load manifest facts requiring OCR
    ocr_facts: list[dict] = []
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                all_facts = json.load(f)
                ocr_facts = [fact for fact in all_facts if fact.get("requires_ocr")]
        except Exception as exc:
            logger.warning("Could not load manifest.json: %s", exc)

    # 3. Target documents to evaluate
    targets = [
        {"file": "WHS-1800_scanned.pdf", "type": "pdf"},
        {"file": "TEX-12_scanned.pdf", "type": "pdf"},
        {"file": "REF-320_spec_plate.png", "type": "image"},
    ]

    print(f"\nTesseract Binary : {settings.TESSERACT_CMD or 'system PATH'}")
    print(f"Confidence Gate  : {settings.OCR_CONF_THRESHOLD * 100:.0f}%")
    print(f"Evaluating Files : {', '.join(t['file'] for t in targets)}\n")

    header = (
        f"{'Document':<24} | {'Pg':<3} | {'Mean Conf':<9} | "
        f"{'Facts Hit':<9} | {'CER':<6} | {'Snippet Preview':<28}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    total_pages = 0
    total_conf = 0.0
    total_facts = 0
    matched_facts = 0

    for target in targets:
        file_path = raw_dir / target["file"]
        if not file_path.exists():
            print(
                f"{target['file']:<24} | {'N/A':<3} | {'MISSING':<9} | "
                f"{'-':<9} | {'-':<6} | {'File not found':<28}"
            )
            continue

        if target["type"] == "pdf":
            try:
                try:
                    import pymupdf as fitz
                except ImportError:
                    import fitz

                pdf = fitz.open(str(file_path))
                for page_idx in range(len(pdf)):
                    page = pdf[page_idx]
                    page_num = page_idx + 1

                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    binary = preprocess(img)
                    ocr_res = engine.run(binary)

                    page_facts = [
                        f
                        for f in ocr_facts
                        if f.get("document") == target["file"]
                        and (f.get("page") == page_num or f.get("page") is None)
                    ]

                    page_matched = 0
                    cers: list[float] = []
                    for fact in page_facts:
                        raw_fact = fact.get("raw") or fact.get("value") or ""
                        search_strs = fact.get("search_strings", [raw_fact])
                        hit = any(s.lower() in ocr_res.text.lower() for s in search_strs if s)
                        if hit:
                            page_matched += 1
                        if raw_fact:
                            sub_text = ocr_res.text[: len(raw_fact)]
                            cer_val = 0.0 if hit else _compute_cer(raw_fact, sub_text)
                            cers.append(cer_val)

                    mean_cer = sum(cers) / len(cers) if cers else 0.0
                    fact_str = f"{page_matched}/{len(page_facts)}" if page_facts else "N/A"

                    snippet = ocr_res.text.replace("\n", " ").strip()[:25]
                    if len(ocr_res.text.replace("\n", " ").strip()) > 25:
                        snippet += ".."

                    print(
                        f"{target['file']:<24} | "
                        f"{page_num:<3} | "
                        f"{ocr_res.mean_confidence * 100:>7.1f}% | "
                        f"{fact_str:<9} | "
                        f"{mean_cer:>5.2f} | "
                        f"{snippet:<28}"
                    )

                    total_pages += 1
                    total_conf += ocr_res.mean_confidence
                    total_facts += len(page_facts)
                    matched_facts += page_matched

            except Exception as exc:
                err_msg = str(exc)[:25]
                print(
                    f"{target['file']:<24} | {'ERR':<3} | {'ERROR':<9} | "
                    f"{'-':<9} | {'-':<6} | {err_msg:<28}"
                )

        elif target["type"] == "image":
            try:
                img = Image.open(file_path)
                binary = preprocess(img)
                ocr_res = engine.run(binary)

                doc_facts = [f for f in ocr_facts if f.get("document") == target["file"]]
                img_matched = 0
                cers = []
                for fact in doc_facts:
                    raw_fact = fact.get("raw") or fact.get("value") or ""
                    search_strs = fact.get("search_strings", [raw_fact])
                    hit = any(s.lower() in ocr_res.text.lower() for s in search_strs if s)
                    if hit:
                        img_matched += 1
                    if raw_fact:
                        sub_text = ocr_res.text[: len(raw_fact)]
                        cer_val = 0.0 if hit else _compute_cer(raw_fact, sub_text)
                        cers.append(cer_val)

                mean_cer = sum(cers) / len(cers) if cers else 0.0
                fact_str = f"{img_matched}/{len(doc_facts)}" if doc_facts else "N/A"
                snippet = ocr_res.text.replace("\n", " ").strip()[:25]
                if len(ocr_res.text.replace("\n", " ").strip()) > 25:
                    snippet += ".."

                print(
                    f"{target['file']:<24} | "
                    f"{'1':<3} | "
                    f"{ocr_res.mean_confidence * 100:>7.1f}% | "
                    f"{fact_str:<9} | "
                    f"{mean_cer:>5.2f} | "
                    f"{snippet:<28}"
                )

                total_pages += 1
                total_conf += ocr_res.mean_confidence
                total_facts += len(doc_facts)
                matched_facts += img_matched

            except Exception as exc:
                err_msg = str(exc)[:25]
                print(
                    f"{target['file']:<24} | {'ERR':<3} | {'ERROR':<9} | "
                    f"{'-':<9} | {'-':<6} | {err_msg:<28}"
                )

    print("-" * len(header))
    if total_pages > 0:
        avg_conf = (total_conf / total_pages) * 100
        recovery_pct = (matched_facts / total_facts * 100) if total_facts else 0.0
        print(f"Summary: Evaluated {total_pages} page(s). Average Confidence: {avg_conf:.1f}%.")
        summary_rec = (
            f"Manifest Ground-Truth Fact Recovery: "
            f"{matched_facts}/{total_facts} ({recovery_pct:.1f}%).\n"
        )
        print(summary_rec)
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(generate_ocr_report())
