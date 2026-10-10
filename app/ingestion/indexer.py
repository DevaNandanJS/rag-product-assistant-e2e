"""Orchestrator for document normalization, chunking strategies,
and idempotent indexing into Qdrant.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.core.schemas import Chunk
from app.ingestion.chunkers.fixed import chunk_fixed
from app.ingestion.chunkers.prose import chunk_prose
from app.ingestion.chunkers.structured import (
    chunk_structured_products,
    chunk_structured_suppliers,
)
from app.ingestion.chunkers.tokens import TokenCounter
from app.ingestion.normalize import DocumentNormalizer
from app.retrieval.store import QdrantStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".json", ".csv", ".md", ".markdown", ".txt", ".pdf"}


@dataclass
class IngestionReport:
    """Summary of an ingestion run across all discovered documents."""

    chunker: str
    total_files: int = 0
    total_chunks: int = 0
    chunks_by_type: dict[str, int] = field(default_factory=dict)
    source_types: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0
    max_chunk_tokens: int = 0
    duration_s: float = 0.0
    documents_indexed: list[str] = field(default_factory=list)
    skipped: bool = False

    def summary_table(self) -> str:
        """Render a formatted ASCII summary table."""
        lines = [
            "=" * 65,
            f"          INGESTION REPORT ({self.chunker.upper()} CHUNKER)",
            "=" * 65,
            f"Total Files Processed   : {self.total_files}",
            f"Total Chunks Generated  : {self.total_chunks}",
            f"Total Tokens Indexed    : {self.total_tokens}",
            f"Max Chunk Tokens        : {self.max_chunk_tokens} (ceiling: 480)",
            f"Execution Duration      : {self.duration_s:.2f}s",
            "-" * 65,
            "Chunk Distribution by Type:",
        ]
        for ctype, count in sorted(self.chunks_by_type.items()):
            lines.append(f"  • {ctype:<15}: {count}")
        lines.append("-" * 65)
        lines.append("Source Type Breakdown:")
        for stype, count in sorted(self.source_types.items()):
            lines.append(f"  • {stype:<15}: {count}")
        lines.append("=" * 65)
        return "\n".join(lines)


class Indexer:
    """Idempotently indexes directory files into Qdrant using the selected chunker."""

    def __init__(
        self,
        normalizer: DocumentNormalizer,
        store: QdrantStore,
        counter: TokenCounter,
        chunker: Literal["structured", "fixed"] = "structured",
        target_tokens: int = 250,
        max_tokens: int = 480,
    ) -> None:
        self.normalizer = normalizer
        self.store = store
        self.counter = counter
        self.chunker = chunker
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens

    def ingest_all(
        self,
        data_dir: Path | str,
        if_empty: bool = False,
        rebuild: bool = False,
    ) -> IngestionReport:
        """Scan data directory, normalize, chunk, and index into Qdrant.

        Args:
            data_dir: Path to directory containing raw files.
            if_empty: If True, skips ingestion if Qdrant collection already has points.
            rebuild: If True, wipes existing collection before indexing.

        Returns:
            IngestionReport with detailed stats.
        """
        start_time = time.perf_counter()
        target_dir = Path(data_dir)
        report = IngestionReport(chunker=self.chunker)

        if rebuild:
            col_name = self.store.collection_name(self.chunker)
            logger.info("Rebuild requested: purging collection '%s'.", col_name)
            self.store.delete_collection(chunker=self.chunker)

        if if_empty:
            count = self.store.count(chunker=self.chunker)
            if count > 0:
                logger.info("Collection '%s' already contains %d points; skipping (--if-empty).",
                            self.store.collection_name(self.chunker), count)
                report.skipped = True
                report.total_chunks = count
                report.duration_s = time.perf_counter() - start_time
                return report

        # Discover supported source files
        all_files: list[Path] = []
        if target_dir.exists():
            for p in target_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    all_files.append(p)
        all_files.sort(key=lambda p: str(p.relative_to(target_dir)))

        for file_path in all_files:
            norm_doc = self.normalizer.normalize(file_path)
            chunks: list[Chunk] = []

            if self.chunker == "structured":
                # Products
                if norm_doc.product_records:
                    chunks.extend(
                        chunk_structured_products(
                            norm_doc.product_records,
                            norm_doc.source_file,
                            self.counter,
                            max_tokens=self.max_tokens,
                        )
                    )

                # Suppliers
                if norm_doc.supplier_records:
                    chunks.extend(
                        chunk_structured_suppliers(
                            norm_doc.supplier_records,
                            norm_doc.source_file,
                            self.counter,
                            max_tokens=self.max_tokens,
                        )
                    )

                # Bulletins
                for b in norm_doc.bulletin_records:
                    p_id = b.related_product_ids[0] if b.related_product_ids else None
                    chunks.extend(
                        chunk_prose(
                            text=b.content,
                            document=norm_doc.source_file,
                            section=b.title,
                            product_id=p_id,
                            counter=self.counter,
                            target_tokens=self.target_tokens,
                            max_tokens=self.max_tokens,
                        )
                    )

                # Text Sections
                for sec in norm_doc.sections:
                    chunks.extend(
                        chunk_prose(
                            text=sec.body,
                            document=norm_doc.source_file,
                            section=sec.heading,
                            counter=self.counter,
                            target_tokens=self.target_tokens,
                            max_tokens=self.max_tokens,
                        )
                    )

                # Normalized Pages (PDFs, OCR)
                for page in norm_doc.pages:
                    chunks.extend(
                        chunk_prose(
                            text=page.text,
                            document=norm_doc.source_file,
                            page=page.page_number,
                            source_type=page.source_type,  # type: ignore[arg-type]
                            ocr_confidence=page.ocr_confidence,
                            suspicious=page.suspicious,
                            counter=self.counter,
                            target_tokens=self.target_tokens,
                            max_tokens=self.max_tokens,
                        )
                    )

            elif self.chunker == "fixed":
                # Fixed 500-token sliding window across all documents (Baseline B0)
                doc_text_parts: list[str] = []
                for prod in norm_doc.product_records:
                    doc_text_parts.append(f"{prod.product_name}\n{prod.description or ''}")
                for sup in norm_doc.supplier_records:
                    sup_text = f"{sup.supplier_name} {sup.headquarters} {sup.lead_time_notes or ''}"
                    doc_text_parts.append(sup_text)
                for b in norm_doc.bulletin_records:
                    doc_text_parts.append(b.content)
                for sec in norm_doc.sections:
                    doc_text_parts.append(sec.body)
                for page in norm_doc.pages:
                    if page.text.strip():
                        doc_text_parts.append(page.text)

                combined_text = "\n\n".join(t for t in doc_text_parts if t.strip())
                has_records = bool(norm_doc.product_records or norm_doc.supplier_records)
                stype: Literal["structured", "extracted", "ocr", "ocr_vision"] = (
                    "structured" if has_records else "extracted"
                )
                chunks = chunk_fixed(
                    text=combined_text,
                    document=norm_doc.source_file,
                    page=None,
                    source_type=stype,
                    ocr_confidence=None,
                    counter=self.counter,
                    window=500,
                    overlap=50,
                )

            # Idempotently update Qdrant for this document
            self.store.replace_document(norm_doc.source_file, chunks, chunker=self.chunker)

            # Accumulate report metrics
            report.total_files += 1
            report.total_chunks += len(chunks)
            report.documents_indexed.append(norm_doc.source_file)

            for c in chunks:
                report.total_tokens += c.token_count
                report.max_chunk_tokens = max(report.max_chunk_tokens, c.token_count)
                report.chunks_by_type[c.chunk_type] = report.chunks_by_type.get(c.chunk_type, 0) + 1
                report.source_types[c.source_type] = report.source_types.get(c.source_type, 0) + 1

        report.duration_s = time.perf_counter() - start_time
        return report
