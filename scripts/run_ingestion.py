"""
SyllAIq — Ingestion Pipeline Runner
====================================
Executes Phase 1 ingestion:
1. Ingests official RGPV OS Syllabus
2. Ingests all authentic RGPV PYQ Papers (2023-2026)
3. Ingests any Textbook PDFs present in data/raw/os/textbook/
4. Outputs standardized data/processed/raw_documents.json
"""

import json
from pathlib import Path
from typing import List

from config.settings import (
    OS_SYLLABUS_PATH,
    OS_PYQS_PATH,
    OS_TEXTBOOK_PATH,
    RAW_DOCUMENTS_JSON,
    DATA_PROCESSED_PATH,
)
from models.documents import Document, SourceType
from ingestion.pdf_loader import PDFLoader
from utils.logger import get_logger

logger = get_logger(__name__)


def run_ingestion_pipeline() -> List[Document]:
    """
    Executes end-to-end ingestion and writes raw_documents.json.
    """
    logger.info("=" * 70)
    logger.info("🚀 Starting SyllAIq Phase 1 Ingestion Pipeline...")
    logger.info("=" * 70)

    loader = PDFLoader(use_cleaner=True)
    all_documents: List[Document] = []

    # -------------------------------------------------------------
    # Step 1: Ingest Syllabus
    # -------------------------------------------------------------
    logger.info("\n[1/3] Ingesting RGPV OS Syllabus...")
    syllabus_json = OS_SYLLABUS_PATH / "rgpv_os_syllabus.json"
    syllabus_txt = OS_SYLLABUS_PATH / "rgpv_os_syllabus.txt"

    if syllabus_json.exists():
        syllabus_docs = loader.load_syllabus_file(syllabus_json)
        all_documents.extend(syllabus_docs)
        logger.info(f"✅ Ingested {len(syllabus_docs)} syllabus unit documents.")
    elif syllabus_txt.exists():
        syllabus_docs = loader.load_syllabus_file(syllabus_txt)
        all_documents.extend(syllabus_docs)
        logger.info(f"✅ Ingested {len(syllabus_docs)} syllabus unit documents.")

    # -------------------------------------------------------------
    # Step 2: Ingest PYQ Papers
    # -------------------------------------------------------------
    logger.info("\n[2/3] Ingesting Authentic RGPV PYQs...")
    pyq_json = OS_PYQS_PATH / "rgpv_os_pyqs_dataset.json"
    if pyq_json.exists():
        pyq_docs = loader.load_pyq_json(pyq_json)
        all_documents.extend(pyq_docs)
        logger.info(f"✅ Ingested {len(pyq_docs)} atomic PYQ question documents.")
    
    # Also check if any PDF PYQ papers exist in pyqs/
    for pdf_file in OS_PYQS_PATH.glob("*.pdf"):
        pyq_pdf_docs = loader.load_pdf(pdf_file, source_type=SourceType.PYQ)
        all_documents.extend(pyq_pdf_docs)
        logger.info(f"✅ Ingested {len(pyq_pdf_docs)} pages from PYQ PDF {pdf_file.name}")

    # -------------------------------------------------------------
    # Step 3: Ingest Textbooks
    # -------------------------------------------------------------
    logger.info("\n[3/3] Ingesting Textbook PDFs...")
    textbook_count = 0
    for pdf_file in OS_TEXTBOOK_PATH.glob("*.pdf"):
        tb_docs = loader.load_pdf(pdf_file, source_type=SourceType.TEXTBOOK, book_title=pdf_file.stem)
        all_documents.extend(tb_docs)
        textbook_count += len(tb_docs)
        logger.info(f"✅ Ingested {len(tb_docs)} textbook pages from {pdf_file.name}")

    if textbook_count == 0:
        logger.info("ℹ️  No textbook PDFs found in data/raw/os/textbook/ yet. (Place Silberschatz/Galvin PDF there anytime).")

    # -------------------------------------------------------------
    # Step 4: Save to data/processed/raw_documents.json
    # -------------------------------------------------------------
    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    serialized_docs = [doc.model_dump() for doc in all_documents]

    with open(RAW_DOCUMENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(serialized_docs, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 70)
    logger.info(f"🎉 Ingestion Pipeline Complete!")
    logger.info(f"📁 Output: {RAW_DOCUMENTS_JSON}")
    logger.info(f"📊 Total Documents Ingested: {len(all_documents)}")
    logger.info("=" * 70 + "\n")

    return all_documents


if __name__ == "__main__":
    run_ingestion_pipeline()
