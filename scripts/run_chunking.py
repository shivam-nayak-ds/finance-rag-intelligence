"""
SyllAIq Chunking Pipeline Runner (Production-Grade)
===================================================
Executes the Phase 2 domain-aware chunking pipeline:
1. Loads raw documents from data/processed/raw_documents.json
2. Routes each document type to its specialized chunker:
   - Textbooks -> CSRecursiveChunker (800 chars, 150 overlap, rich metadata)
   - PYQs -> PYQChunker (1 Question = 1 Atomic Chunk)
   - Syllabus -> SyllabusChunker (1 Unit = 1 Chunk)
3. Atomically saves output to data/processed/chunks.json
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from chunking.pyq_chunker import PYQChunker
from chunking.recursive_chunker import CSRecursiveChunker
from chunking.syllabus_chunker import SyllabusChunker
from config.settings import (
    CHUNKS_JSON,
    DATA_PROCESSED_PATH,
    RAW_DOCUMENTS_JSON,
    TEXTBOOK_CHUNK_OVERLAP,
    TEXTBOOK_CHUNK_SIZE,
)
from models.documents import Document, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkingReport:
    """Execution summary report for Phase 2 chunking."""
    total_raw_documents: int
    total_chunks_created: int
    textbook_chunks: int
    pyq_chunks: int
    syllabus_chunks: int
    average_chunk_length: float
    duration_seconds: float
    output_file: str


def run_chunking_pipeline() -> ChunkingReport:
    """
    Executes the chunking pipeline and writes data/processed/chunks.json.

    Returns:
        ChunkingReport with execution metrics.
    """
    start_time = time.perf_counter()
    logger.info("=" * 70)
    logger.info("🚀 Starting SyllAIq Domain-Aware Chunking Pipeline...")
    logger.info("=" * 70)

    if not RAW_DOCUMENTS_JSON.exists():
        raise FileNotFoundError(
            f"Input file '{RAW_DOCUMENTS_JSON}' not found. Please run 'scripts/run_ingestion.py' first."
        )

    # 1. Load raw documents
    logger.info(f"Loading raw documents from '{RAW_DOCUMENTS_JSON}'...")
    with open(RAW_DOCUMENTS_JSON, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_docs = [Document(**item) for item in raw_data]
    logger.info(f"Loaded {len(raw_docs)} raw documents.")

    # 2. Instantiate domain-specific chunkers
    textbook_chunker = CSRecursiveChunker(
        chunk_size=TEXTBOOK_CHUNK_SIZE,
        chunk_overlap=TEXTBOOK_CHUNK_OVERLAP,
    )
    pyq_chunker = PYQChunker()
    syllabus_chunker = SyllabusChunker()

    all_chunks: List[Document] = []
    textbook_chunk_count = 0
    pyq_chunk_count = 0
    syllabus_chunk_count = 0

    # 3. Route documents by source type
    logger.info("\nProcessing documents with specialized chunking strategies...")
    for doc in raw_docs:
        if doc.source_type == SourceType.TEXTBOOK:
            chunks = textbook_chunker.split_document(doc)
            all_chunks.extend(chunks)
            textbook_chunk_count += len(chunks)
        elif doc.source_type == SourceType.PYQ:
            chunks = pyq_chunker.split_document(doc)
            all_chunks.extend(chunks)
            pyq_chunk_count += len(chunks)
        elif doc.source_type == SourceType.SYLLABUS:
            chunks = syllabus_chunker.split_document(doc)
            all_chunks.extend(chunks)
            syllabus_chunk_count += len(chunks)
        else:
            chunks = textbook_chunker.split_document(doc)
            all_chunks.extend(chunks)

    # Calculate average chunk length
    total_chars = sum(len(c.text) for c in all_chunks)
    avg_length = round(total_chars / len(all_chunks), 2) if all_chunks else 0.0

    # 4. Atomic file persistence
    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    temp_output = CHUNKS_JSON.with_suffix(".json.tmp")
    serialized = [chunk.model_dump() for chunk in all_chunks]

    try:
        with open(temp_output, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

        if os.name == "nt" and CHUNKS_JSON.exists():
            CHUNKS_JSON.unlink()
        temp_output.rename(CHUNKS_JSON)
        logger.info(f"✅ Atomically wrote {len(all_chunks)} chunks to '{CHUNKS_JSON}'")
    except Exception as io_err:
        logger.critical(f"Failed writing chunks atomically: {io_err}")
        if temp_output.exists():
            temp_output.unlink()
        raise

    duration = round(time.perf_counter() - start_time, 2)

    report = ChunkingReport(
        total_raw_documents=len(raw_docs),
        total_chunks_created=len(all_chunks),
        textbook_chunks=textbook_chunk_count,
        pyq_chunks=pyq_chunk_count,
        syllabus_chunks=syllabus_chunk_count,
        average_chunk_length=avg_length,
        duration_seconds=duration,
        output_file=str(CHUNKS_JSON),
    )

    logger.info("\n" + "=" * 70)
    logger.info("🎉 Chunking Pipeline Completed Successfully!")
    logger.info(f"   • Raw Documents Read : {report.total_raw_documents}")
    logger.info(f"   • Total Chunks Made  : {report.total_chunks_created}")
    logger.info(f"   • Textbook Chunks    : {report.textbook_chunks} (Galvin 10th Ed)")
    logger.info(f"   • PYQ Chunks         : {report.pyq_chunks} (Atomic Questions)")
    logger.info(f"   • Syllabus Chunks    : {report.syllabus_chunks} (Unit Boundaries)")
    logger.info(f"   • Avg Chunk Length   : {report.average_chunk_length} characters")
    logger.info(f"   • Duration           : {report.duration_seconds}s")
    logger.info("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    run_chunking_pipeline()
