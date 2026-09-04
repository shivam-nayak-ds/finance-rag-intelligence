"""Ingestion pipeline script for loading raw documents into standardized JSON format."""

import json
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from typing import List

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
    """Summary metrics of the ingestion run."""
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
    """Executes ingestion across syllabus, PYQ, and textbook sources."""
    start_time = time.perf_counter()
    logger.info("Starting document ingestion pipeline")

    loader = PDFLoader(clean_extracted_text=True)
    all_documents: List[Document] = []
    syllabus_count = 0
    pyq_count = 0
    textbook_count = 0

    # 1. Ingest Syllabus
    logger.info("Ingesting syllabus documents from: %s", OS_SYLLABUS_PATH)
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
                logger.error("Failed parsing syllabus file '%s': %s", syllabus_path, err)

    # 2. Ingest PYQs
    logger.info("Ingesting PYQ dataset from: %s", OS_PYQS_PATH)
    pyq_json = OS_PYQS_PATH / "rgpv_os_pyqs_dataset.json"
    if pyq_json.exists():
        try:
            p_docs = loader.load_pyq_json(pyq_json)
            all_documents.extend(p_docs)
            pyq_count = len(p_docs)
        except Exception as err:
            logger.error("Failed parsing PYQ dataset '%s': %s", pyq_json, err)

    # 3. Ingest Textbook PDFs
    logger.info("Ingesting textbook PDFs from: %s", OS_TEXTBOOK_PATH)
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
            logger.error("Error loading textbook '%s': %s", pdf_path.name, err)

    # 4. Atomic File Write
    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    temp_output_path = RAW_DOCUMENTS_JSON.with_suffix(".json.tmp")
    serialized = [doc.model_dump() for doc in all_documents]

    try:
        with open(temp_output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

        if os.name == "nt" and RAW_DOCUMENTS_JSON.exists():
            RAW_DOCUMENTS_JSON.unlink()
        temp_output_path.rename(RAW_DOCUMENTS_JSON)
        logger.info("Wrote %d documents to '%s'", len(all_documents), RAW_DOCUMENTS_JSON)
    except Exception as io_err:
        logger.critical("Atomic file write failed: %s", io_err)
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

    logger.info(
        "Ingestion completed: %d total docs (%d textbook pages, %d PYQs, %d syllabus units) in %.2fs",
        report.total_documents,
        report.textbook_documents,
        report.pyq_documents,
        report.syllabus_documents,
        report.duration_seconds,
    )

    return report


if __name__ == "__main__":
    run_ingestion_pipeline()
