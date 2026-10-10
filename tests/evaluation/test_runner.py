"""Unit tests for EvalRunner and report formatting using in-memory store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.schemas import Chunk
from app.evaluation.report import (
    format_comparison_table,
    format_markdown_table,
    save_result,
)
from app.evaluation.runner import EvalConfig, EvalRunner
from app.retrieval.embedder import FakeEmbedder
from app.retrieval.store import QdrantStore


@pytest.fixture
def test_env(tmp_path: Path):
    settings = Settings(
        APP_ENV="test",
        QDRANT_MODE="memory",
        COLLECTION_PREFIX="test_eval",
        CHUNKER="fixed",
        EMBEDDING_MODEL="fake-model",
    )
    embedder = FakeEmbedder(dim=16)
    store = QdrantStore(settings=settings, embedder=embedder)

    # Ingest a sample chunk into fixed collection
    sample_chunk = Chunk(
        chunk_id="PKG-120_CHUNK_001",
        point_id="00000000-0000-0000-0000-000000000001",
        text="CartonPro 1200 Semi-Automatic Carton Sealer. Throughput: 18 ctn/min.",
        display_text="CartonPro 1200 Semi-Automatic Carton Sealer. Throughput: 18 ctn/min.",
        chunk_type="fixed",
        document="catalog.json",
        source_type="structured",
        token_count=12,
    )
    store.replace_document("catalog.json", [sample_chunk], chunker="fixed")

    # Create a small questions.json
    questions_file = tmp_path / "questions.json"
    questions_data = [
        {
            "id": "q01",
            "split": "dev",
            "category": "direct_factual",
            "question": "What is the maximum throughput of the PKG-120?",
            "answerable": True,
            "reference_answer": "18 ctn/min",
            "evidence": [
                {
                    "document": "catalog.json",
                    "page": None,
                    "key_fact": "18 ctn/min",
                }
            ],
        },
        {
            "id": "q02",
            "split": "dev",
            "category": "unanswerable_offtopic",
            "question": "What is the range of SkyDrone 500?",
            "answerable": False,
            "reference_answer": "Not documented",
            "evidence": [],
        },
    ]
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions_data, f)

    return settings, embedder, store, questions_file


def test_eval_runner_execution(test_env, tmp_path: Path) -> None:
    settings, embedder, store, questions_file = test_env

    config = EvalConfig(
        run_id="TEST_RUN",
        description="Test eval run",
        chunker="fixed",
        retrieval_mode="dense",
        top_k=5,
        split="dev",
    )

    runner = EvalRunner(settings=settings, embedder=embedder, store=store)
    result = runner.run(config, questions_path=questions_file)

    assert result.run_id == "TEST_RUN"
    assert len(result.per_query) == 2
    assert result.overall.count == 1  # 1 answerable query
    assert result.overall.mean_recall_at_k == 1.0
    assert result.overall.hit_rate == 1.0

    # Test report formatting
    report_md = format_markdown_table(result)
    assert "TEST_RUN" in report_md
    assert "direct_factual" in report_md

    comparison_md = format_comparison_table({"TEST_RUN": result})
    assert "Benchmark Ablation Comparison" in comparison_md
    assert "TEST_RUN" in comparison_md

    # Test save
    out_file = tmp_path / "result.json"
    save_result(result, out_file)
    assert out_file.exists()
