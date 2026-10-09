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

    # Command: ingest (stub)
    ingest_parser = subparsers.add_parser("ingest", help="Ingest catalog and documentation")
    ingest_parser.add_argument("--chunker", choices=["structured", "fixed"], default="structured")
    ingest_parser.add_argument("--if-empty", action="store_true")
    ingest_parser.add_argument("--rebuild", action="store_true")

    # Command: eval (stub)
    eval_parser = subparsers.add_parser("eval", help="Run evaluation benchmark")
    eval_parser.add_argument("--run", type=str, default="eval/runs/B0.yaml")

    # Command: serve (stub)
    subparsers.add_parser("serve", help="Start FastAPI Uvicorn server")

    parsed, remaining = parser.parse_known_args()

    if parsed.command == "check":
        sys.exit(run_check())
    elif parsed.command == "test":
        sys.exit(run_test(parsed.pytest_args + remaining))
    elif parsed.command in ("ingest", "eval", "serve"):
        print(f"Command '{parsed.command}' will be registered in its respective build phase.")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
