"""
SyllAIq Ingestion Runner (Production-Grade)
===========================================
Orchestrates end-to-end ingestion with atomic file persistence,
detailed performance timing, and structured reporting.
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DATA_PROCESSED_PATH,
    OS_PYQS_PATH,
    OS_SYLLABUS_PATH,
    OS_TEXTBOOK_PATH,
    RAW_DOCUMENTS_JSON,
)
from ingestion.exceptions import IngestionError
from ingestion.pdf_loader import PDFLoader
from models.documents import Document, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionReport:
    """Detailed summary report for pipeline observability."""
    total_documents: int
    syllabus_documents: int
    pyq_documents: int
    textbook_documents: int
    total_pages_scanned: int
    duration_seconds: float
    output_file: str
    status: str
    error_count: int


def run_ingestion_pipeline() -> IngestionReport:
    """
    Executes the ingestion pipeline with atomic persistence and error boundaries.

    Returns:
        IngestionReport with comprehensive execution metrics.
    """
    start_time = time.perf_counter()
    logger.info("=" * 70)
    logger.info("🚀 Starting Production Ingestion Pipeline...")
    logger.info("=" * 70)

    loader = PDFLoader(clean_extracted_text=True)
    all_documents: List[Document] = []
    syllabus_count = 0
    pyq_count = 0
    textbook_count = 0

    # -------------------------------------------------------------
    # 1. Ingest Syllabus
    # -------------------------------------------------------------
    logger.info("\n[1/3] Ingesting RGPV OS Syllabus...")
    syllabus_candidates = [
        OS_SYLLABUS_PATH / "rgpv_os_syllabus.json",
        OS_SYLLABUS_PATH / "rgpv_os_syllabus.md",
        OS_SYLLABUS_PATH / "rgpv_os_syllabus.txt",
    ]

    for syllabus_path in syllabus_candidates:
        if syllabus_path.exists():
            try:
                s_docs = loader.load_syllabus(syllabus_path)
                all_documents.extend(s_docs)
                syllabus_count = len(s_docs)
                break
            except Exception as err:
                logger.error(f"Failed parsing syllabus at {syllabus_path}: {err}")

    # -------------------------------------------------------------
    # 2. Ingest Authentic PYQs
    # -------------------------------------------------------------
    logger.info("\n[2/3] Ingesting Authentic RGPV PYQ Dataset...")
    pyq_json = OS_PYQS_PATH / "rgpv_os_pyqs_dataset.json"
    if pyq_json.exists():
        try:
            p_docs = loader.load_pyq_json(pyq_json)
            all_documents.extend(p_docs)
            pyq_count = len(p_docs)
        except Exception as err:
            logger.error(f"Failed parsing PYQ dataset at {pyq_json}: {err}")

    # -------------------------------------------------------------
    # 3. Ingest Textbook PDFs
    # -------------------------------------------------------------
    logger.info("\n[3/3] Ingesting Textbook PDFs from data/raw/os/textbook/...")
    textbook_files = list(OS_TEXTBOOK_PATH.glob("*.pdf"))

    for pdf_path in textbook_files:
        try:
            tb_docs = loader.load_pdf(
                file_path=pdf_path,
                source_type=SourceType.TEXTBOOK,
                book_title="Operating System Concepts (Galvin 10th Ed)",
            )
            all_documents.extend(tb_docs)
            textbook_count += len(tb_docs)
        except Exception as err:
            logger.error(f"Error loading textbook '{pdf_path.name}': {err}")

    # -------------------------------------------------------------
    # 4. Atomic File Write (Write to .tmp then atomic rename)
    # -------------------------------------------------------------
    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    temp_output_path = RAW_DOCUMENTS_JSON.with_suffix(".json.tmp")

    serialized = [doc.model_dump() for doc in all_documents]

    try:
        with open(temp_output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

        # Atomic file replacement
        if os.name == "nt" and RAW_DOCUMENTS_JSON.exists():
            RAW_DOCUMENTS_JSON.unlink()
        temp_output_path.rename(RAW_DOCUMENTS_JSON)
        logger.info(f"✅ Atomically wrote {len(all_documents)} documents to '{RAW_DOCUMENTS_JSON}'")
    except Exception as io_err:
        logger.critical(f"Failed to write output JSON atomically: {io_err}")
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise IngestionError(f"Atomic file write failed: {io_err}") from io_err

    duration = time.perf_counter() - start_time

    report = IngestionReport(
        total_documents=len(all_documents),
        syllabus_documents=syllabus_count,
        pyq_documents=pyq_count,
        textbook_documents=textbook_count,
        total_pages_scanned=loader.stats.total_pages_scanned,
        duration_seconds=round(duration, 2),
        output_file=str(RAW_DOCUMENTS_JSON),
        status="SUCCESS",
        error_count=len(loader.stats.errors),
    )

    logger.info("\n" + "=" * 70)
    logger.info(f"🎉 Ingestion Report:")
    logger.info(f"   • Total Documents : {report.total_documents}")
    logger.info(f"   • Syllabus Units  : {report.syllabus_documents}")
    logger.info(f"   • PYQ Items       : {report.pyq_documents}")
    logger.info(f"   • Textbook Pages  : {report.textbook_documents}")
    logger.info(f"   • Duration        : {report.duration_seconds}s")
    logger.info(f"   • Errors Logged   : {report.error_count}")
    logger.info("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    run_ingestion_pipeline()
