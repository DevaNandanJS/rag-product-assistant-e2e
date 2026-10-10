"""Unified CLI runner for Filumart RAG Product Assistant.
Supports: check, test, ingest, eval, serve.
"""

import argparse
import platform
import shutil
import sys
from pathlib import Path

import httpx

from app.core.config import ConfigurationError, get_settings


def run_check() -> int:
    """Performs a comprehensive environment and readiness audit."""
    print("=" * 65)
    print("      FILUMART RAG ASSISTANT - READINESS CHECK REPORT")
    print("=" * 65)

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"\n[CRITICAL CONFIG ERROR] {exc.message}")
        return 1
    except Exception as exc:
        print(f"\n[UNEXPECTED STARTUP ERROR] {exc}")
        return 1

    # 1. Python Environment
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    status_py = "[OK]" if py_ok else "[WARN]"
    print(f"{status_py} Python Version: {py_ver} ({platform.platform()})")
    if not py_ok:
        print("     Warning: Python 3.11+ is strongly recommended.")

    # 2. Tesseract OCR Discovery
    tess_path = settings.TESSERACT_CMD
    standard_win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    tess_found = False
    resolved_tess = None

    if tess_path and Path(tess_path).exists():
        tess_found = True
        resolved_tess = tess_path
    elif standard_win_path.exists():
        tess_found = True
        resolved_tess = str(standard_win_path)
    elif shutil.which("tesseract"):
        tess_found = True
        resolved_tess = shutil.which("tesseract")

    if tess_found:
        print(f"[OK] Tesseract OCR: Found at '{resolved_tess}'")
    else:
        print("[DEGRADED] Tesseract OCR: Not found locally.")
        print("           Scanned images will use vision fallback or graceful text degradation.")

    # 3. Model Cache Status
    cache_path = Path(settings.FASTEMBED_CACHE_PATH)
    cache_exists = cache_path.exists()
    print(
        f"[{'OK' if cache_exists else 'INFO'}] FastEmbed Cache: '{settings.FASTEMBED_CACHE_PATH}' "
        f"({'cached' if cache_exists else 'will download on first ingest/test'})"
    )

    # 4. Qdrant Vector Store Accessibility
    qdrant_mode = settings.QDRANT_MODE
    if qdrant_mode == "memory":
        print("[OK] Qdrant DB: Mode 'memory' active (offline ephemeral).")
    elif qdrant_mode == "local":
        local_path = Path(settings.QDRANT_PATH)
        print(f"[OK] Qdrant DB: Mode 'local' active (storage: '{local_path.resolve()}').")
    elif qdrant_mode == "server":
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{settings.QDRANT_URL.rstrip('/')}/readyz")
                if res.status_code == 200:
                    print(f"[OK] Qdrant DB: Server online at {settings.QDRANT_URL}")
                else:
                    print(f"[DEGRADED] Qdrant DB: Server responded with status {res.status_code}")
        except Exception:
            print(f"[DEGRADED] Qdrant DB: Server not reachable at {settings.QDRANT_URL}")
            print(
                "           Use QDRANT_MODE=local or QDRANT_MODE=memory for standalone execution."
            )

    # 5. LLM Credentials & Mode
    active_keys = []
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY not in ("", "your_key_here"):
        active_keys.append("Gemini")
    if settings.GROQ_API_KEY and settings.GROQ_API_KEY not in ("", "your_key_here"):
        active_keys.append("Groq")
    if settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_KEY not in ("", "your_key_here"):
        active_keys.append("OpenRouter")

    if active_keys:
        print(f"[OK] LLM Providers: Active keys configured for {', '.join(active_keys)}")
        print(f"     Generation Mode: {settings.LLM_MODE}")
    else:
        print("[INFO] LLM Providers: No live API keys configured.")
        print("       System operating in Reviewer Mode (LLM_MODE=retrieval_only fallback).")

    print("=" * 65)
    print("Readiness check complete.\n")
    return 0


def run_test(args: list[str]) -> int:
    """Executes pytest suite with provided arguments."""
    import pytest

    pytest_args = args or ["tests", "-v"]
    print(f"Running pytest with args: {pytest_args}")
    return pytest.main(pytest_args)


def run_ingest(args: argparse.Namespace) -> int:
    """Runs ingestion pipeline over DATA_DIR using selected chunker."""
    from app.core.config import get_settings
    from app.ingestion.chunkers.tokens import get_token_counter
    from app.ingestion.indexer import Indexer
    from app.ingestion.normalize import DocumentNormalizer
    from app.retrieval.embedder import FastEmbedEmbedder
    from app.retrieval.store import QdrantStore

    settings = get_settings()

    # Initialize OCR engine if available
    ocr_engine = None
    try:
        from app.ingestion.ocr.tesseract import TesseractEngine

        ocr_engine = TesseractEngine(
            cmd_path=settings.TESSERACT_CMD or "",
            lang=settings.OCR_LANG,
        )
    except Exception as exc:
        print(f"[INFO] Tesseract engine not active ({exc}); using fallback/native text extraction.")

    normalizer = DocumentNormalizer(
        ocr_engine=ocr_engine,
        ocr_dpi=settings.OCR_DPI,
        ocr_conf_threshold=settings.OCR_CONF_THRESHOLD,
        ocr_min_text_chars=settings.OCR_MIN_TEXT_CHARS,
    )

    embedder = FastEmbedEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        cache_dir=settings.FASTEMBED_CACHE_PATH,
    )
    store = QdrantStore(settings=settings, embedder=embedder)
    counter = get_token_counter(
        model_name=settings.EMBEDDING_MODEL,
        cache_dir=settings.FASTEMBED_CACHE_PATH,
    )

    indexer = Indexer(
        normalizer=normalizer,
        store=store,
        counter=counter,
        chunker=args.chunker,
        target_tokens=settings.CHUNK_TARGET_TOKENS,
        max_tokens=480,
    )

    print(f"\nStarting ingestion with chunker '{args.chunker}' over '{settings.DATA_DIR}'...")
    try:
        report = indexer.ingest_all(
            data_dir=settings.DATA_DIR,
            if_empty=args.if_empty,
            rebuild=args.rebuild,
        )
        print(report.summary_table())
        return 0
    except Exception as exc:
        from app.core.errors import VectorDBError

        if isinstance(exc, VectorDBError):
            print(f"\n[VECTOR DB ERROR] {exc.message}")
            if settings.QDRANT_MODE == "server":
                print("Tip: Qdrant server is unreachable at " + settings.QDRANT_URL)
                print(
                    "     Set QDRANT_MODE=local or QDRANT_MODE=memory in .env "
                    "to run without Docker."
                )
            return 1
        raise


def run_eval(args: argparse.Namespace) -> int:
    from pathlib import Path

    from app.core.config import get_settings
    from app.evaluation.report import format_markdown_table, save_result
    from app.evaluation.runner import EvalConfig, EvalRunner
    from app.retrieval.embedder import FastEmbedEmbedder
    from app.retrieval.store import QdrantStore

    settings = get_settings()
    run_file = Path(args.run)
    if not run_file.exists():
        print(f"[ERROR] Run configuration file '{args.run}' does not exist.")
        return 1

    config = EvalConfig.from_yaml(run_file)
    embedder = FastEmbedEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        cache_dir=settings.FASTEMBED_CACHE_PATH,
    )
    store = QdrantStore(settings=settings, embedder=embedder)

    split_target = getattr(args, "split", None) or config.split
    print(f"\nStarting benchmark run '{config.run_id}' (split: {split_target})...")

    runner = EvalRunner(settings=settings, embedder=embedder, store=store)
    result = runner.run(
        config=config,
        questions_path=getattr(args, "questions", "eval/questions.json"),
        split_override=split_target,
    )

    print("\n" + format_markdown_table(result))

    out_path = getattr(args, "output", None) or f"eval/results/{config.run_id}.json"
    saved = save_result(result, out_path)
    print(f"\n[INFO] Benchmark results saved to '{saved}'")
    return 0


def run_eval_report(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from app.evaluation.metrics import AggregateMetrics
    from app.evaluation.report import format_comparison_table
    from app.evaluation.runner import EvalConfig, EvalResult

    results_dir = Path(getattr(args, "results_dir", "eval/results"))
    if not results_dir.exists():
        print(f"[ERROR] Directory '{results_dir}' does not exist.")
        return 1

    results: dict[str, EvalResult] = {}
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            config = EvalConfig(**data["config"])
            ov = data["overall"]
            overall = AggregateMetrics(
                mean_recall_at_k=ov["mean_recall_at_k"],
                mrr=ov["mrr"],
                hit_rate=ov["hit_rate"],
                mean_precision_at_k=ov["mean_precision_at_k"],
                ndcg_at_k=ov["ndcg_at_k"],
                count=ov["count"],
            )
            result = EvalResult(
                run_id=data["run_id"],
                config=config,
                split=data["split"],
                overall=overall,
                by_category={},
                per_query=[],
                retrieved_chunk_ids={},
            )
            results[data["run_id"]] = result
        except Exception as exc:
            print(f"[WARN] Could not parse '{json_file}': {exc}")

    if not results:
        print(f"[INFO] No evaluation result JSON files found in '{results_dir}'.")
        return 0

    print("\n" + format_comparison_table(results))
    return 0


def run_eval_gate(args: argparse.Namespace) -> int:
    """Sweep candidate similarity thresholds on dev split to find optimal gating threshold."""
    from app.core.config import get_settings
    from app.evaluation.runner import EvalRunner
    from app.retrieval.embedder import FastEmbedEmbedder
    from app.retrieval.store import QdrantStore

    settings = get_settings()
    embedder = FastEmbedEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        cache_dir=settings.FASTEMBED_CACHE_PATH,
    )
    store = QdrantStore(settings=settings, embedder=embedder)

    min_thresh = getattr(args, "min_thresh", 0.30)
    max_thresh = getattr(args, "max_thresh", 0.70)
    step = getattr(args, "step", 0.05)
    split_target = getattr(args, "split", "dev")
    chunker = getattr(args, "chunker", "structured")
    questions_path = getattr(args, "questions", "eval/questions.json")

    runner = EvalRunner(settings=settings, embedder=embedder, store=store)
    questions = runner.load_questions(questions_path, split=split_target)

    if not questions:
        print(f"[ERROR] No questions found for split '{split_target}' in '{questions_path}'.")
        return 1

    print(
        f"\nStarting threshold sweep on {len(questions)} questions "
        f"(split: {split_target}, chunker: {chunker})..."
    )
    print(f"Sweeping range [{min_thresh:.2f}, {max_thresh:.2f}] with step {step:.2f}...\n")

    question_scores: list[tuple[bool, float, str]] = []
    for q in questions:
        query_vec = embedder.embed([q.question])[0]
        results = store.dense_search_with_scores(
            query_vector=query_vec,
            top_k=1,
            chunker=chunker,
        )
        top_score = results[0][1] if results else 0.0
        question_scores.append((q.answerable, top_score, q.id))

    thresholds: list[float] = []
    curr = min_thresh
    while curr <= max_thresh + 1e-5:
        thresholds.append(round(curr, 3))
        curr += step

    rows: list[dict[str, str | int]] = []
    best_thresh = min_thresh
    best_objective = -999.0

    for t in thresholds:
        tp = 0  # answerable & passed
        fn = 0  # answerable & rejected (false rejection)
        tn = 0  # unanswerable & rejected (correct catch)
        fp = 0  # unanswerable & passed (false pass)

        for ans, sc, _ in question_scores:
            passed = sc >= t
            if ans:
                if passed:
                    tp += 1
                else:
                    fn += 1
            else:
                if not passed:
                    tn += 1
                else:
                    fp += 1

        total_ans = tp + fn
        total_unans = tn + fp
        ans_pass_rate = (tp / total_ans * 100.0) if total_ans > 0 else 0.0
        unans_reject_rate = (tn / total_unans * 100.0) if total_unans > 0 else 0.0

        # Objective: strongly preserve answerable pass rate while penalizing false rejections
        objective = ans_pass_rate + (0.5 * unans_reject_rate) - (fn * 10.0)
        if objective > best_objective:
            best_objective = objective
            best_thresh = t

        rows.append(
            {
                "threshold": f"{t:.2f}",
                "ans_pass": f"{tp}/{total_ans} ({ans_pass_rate:.1f}%)",
                "unans_reject": f"{tn}/{total_unans} ({unans_reject_rate:.1f}%)",
                "false_rejections": fn,
                "false_passes": fp,
            }
        )

    lines = [
        (
            "| Threshold | Answerable Pass Rate | Off-Topic Rejection Rate "
            "| False Rejections | False Passes |"
        ),
        "|:---|:---|:---|:---|:---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']} | {r['ans_pass']} | {r['unans_reject']} | "
            f"{r['false_rejections']} | {r['false_passes']} |"
        )

    print("\n".join(lines))
    print(
        f"\n[RECOMMENDATION] Optimal threshold: {best_thresh:.2f} "
        f"(minimizes false rejections while catching off-topic queries)."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="filumart",
        description="Filumart RAG Product Assistant CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: check
    subparsers.add_parser("check", help="Verify environment, dependencies, and vector DB")

    # Command: test
    test_parser = subparsers.add_parser("test", help="Run pytest test suite")
    test_parser.add_argument("pytest_args", nargs="*", help="Arguments forwarded to pytest")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest catalog and documentation")
    ingest_parser.add_argument("--chunker", choices=["structured", "fixed"], default="structured")
    ingest_parser.add_argument("--if-empty", action="store_true")
    ingest_parser.add_argument("--rebuild", action="store_true")

    # Command: eval
    eval_parser = subparsers.add_parser("eval", help="Run evaluation benchmark")
    eval_parser.add_argument(
        "--run", type=str, default="eval/runs/B0.yaml", help="Path to run YAML"
    )
    eval_parser.add_argument(
        "--split", choices=["dev", "holdout", "all"], default=None, help="Question split"
    )
    eval_parser.add_argument(
        "--questions", type=str, default="eval/questions.json", help="Questions JSON path"
    )
    eval_parser.add_argument(
        "--output", type=str, default=None, help="Custom output JSON path"
    )

    # Command: eval-report
    report_parser = subparsers.add_parser(
        "eval-report", help="Generate comparative ablation report"
    )
    report_parser.add_argument(
        "--results-dir",
        type=str,
        default="eval/results",
        help="Directory containing run JSONs",
    )

    # Command: eval-gate
    gate_parser = subparsers.add_parser(
        "eval-gate", help="Sweep candidate thresholds on dev split"
    )
    gate_parser.add_argument(
        "--min-thresh", type=float, default=0.30, help="Minimum threshold to test"
    )
    gate_parser.add_argument(
        "--max-thresh", type=float, default=0.70, help="Maximum threshold to test"
    )
    gate_parser.add_argument(
        "--step", type=float, default=0.05, help="Step increment"
    )
    gate_parser.add_argument(
        "--split", choices=["dev", "holdout", "all"], default="dev", help="Question split"
    )
    gate_parser.add_argument(
        "--chunker",
        choices=["structured", "fixed"],
        default="structured",
        help="Chunker type",
    )
    gate_parser.add_argument(
        "--questions", type=str, default="eval/questions.json", help="Questions JSON path"
    )

    # Command: serve (stub)
    subparsers.add_parser("serve", help="Start FastAPI Uvicorn server")

    parsed, remaining = parser.parse_known_args()

    if parsed.command == "check":
        sys.exit(run_check())
    elif parsed.command == "test":
        sys.exit(run_test(parsed.pytest_args + remaining))
    elif parsed.command == "ingest":
        sys.exit(run_ingest(parsed))
    elif parsed.command == "eval":
        sys.exit(run_eval(parsed))
    elif parsed.command == "eval-report":
        sys.exit(run_eval_report(parsed))
    elif parsed.command == "eval-gate":
        sys.exit(run_eval_gate(parsed))
    elif parsed.command == "serve":
        print(f"Command '{parsed.command}' will be registered in its respective build phase.")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

