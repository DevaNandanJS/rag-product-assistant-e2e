"""Markdown table generation and result serialization for evaluation runs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from app.evaluation.runner import EvalResult


def format_markdown_table(result: EvalResult) -> str:
    """Format an EvalResult as a readable GitHub-flavored Markdown report."""
    cfg_line = (
        f"*Configuration:* Chunker: `{result.config.chunker}`, "
        f"Mode: `{result.config.retrieval_mode}`, Top-K: `{result.config.top_k}`"
    )
    lines: list[str] = [
        f"## Evaluation Run: `{result.run_id}` (Split: `{result.split}`)",
        f"*Description:* {result.config.description or 'N/A'}",
        cfg_line,
        "",
        f"### Overall Retrieval Metrics (Answerable Queries: {result.overall.count})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Recall@{result.config.top_k}** | {result.overall.mean_recall_at_k:.4f} |",
        f"| **MRR** | {result.overall.mrr:.4f} |",
        f"| **Hit Rate@{result.config.top_k}** | {result.overall.hit_rate:.4f} |",
        f"| **Precision@{result.config.top_k}** | {result.overall.mean_precision_at_k:.4f} |",
        f"| **nDCG@{result.config.top_k}** | {result.overall.ndcg_at_k:.4f} |",
        "",
        "### Category Breakdown",
        "",
        "| Category | Queries | Recall@K | MRR | Hit Rate | Precision@K | nDCG@K |",
        "|---|---|---|---|---|---|---|",
    ]

    for cat, agg in sorted(result.by_category.items()):
        lines.append(
            f"| `{cat}` | {agg.count} | {agg.mean_recall_at_k:.4f} | {agg.mrr:.4f} | "
            f"{agg.hit_rate:.4f} | {agg.mean_precision_at_k:.4f} | {agg.ndcg_at_k:.4f} |"
        )

    return "\n".join(lines)


def format_comparison_table(results: Mapping[str, EvalResult]) -> str:
    """Format a multi-run comparison table for ablations (e.g., B0 through B4)."""
    table_header = (
        "| Run ID | Chunker | Mode | Top-K | Recall@K | MRR | "
        "Hit Rate | Precision@K | nDCG@K |"
    )
    lines: list[str] = [
        "## Benchmark Ablation Comparison",
        "",
        table_header,
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for run_id, res in sorted(results.items()):
        cfg = res.config
        ov = res.overall
        lines.append(
            f"| **{run_id}** | `{cfg.chunker}` | `{cfg.retrieval_mode}` | {cfg.top_k} | "
            f"**{ov.mean_recall_at_k:.4f}** | **{ov.mrr:.4f}** | {ov.hit_rate:.4f} | "
            f"{ov.mean_precision_at_k:.4f} | {ov.ndcg_at_k:.4f} |"
        )

    return "\n".join(lines)


def save_result(result: EvalResult, output_path: str | Path) -> Path:
    """Serialize an EvalResult to a formatted JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result.as_dict(), f, indent=2)
    return out
