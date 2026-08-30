"""
SyllAIq — PDF & Document Loader
================================
Loads academic PDFs, textbooks, syllabus files, and PYQ datasets.
Extracts text page-by-page and enriches each document with source metadata.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union

from models.documents import Document, SourceType
from ingestion.data_cleaner import DataCleaner
from ingestion.metadata_tagger import MetadataTagger
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoader:
    """
    High-performance PDF and document loader using PyMuPDF (fitz)
    with fallbacks to pdfplumber and pypdf.
    """

    def __init__(self, use_cleaner: bool = True):
        self.use_cleaner = use_cleaner
        self.cleaner = DataCleaner()
        self.tagger = MetadataTagger()

    def load_pdf(
        self,
        file_path: Union[str, Path],
        source_type: SourceType = SourceType.TEXTBOOK,
        book_title: Optional[str] = None,
        chapter: Optional[int] = None,
    ) -> List[Document]:
        """
        Extracts text from a PDF file page by page into a list of Document objects.

        Args:
            file_path: Path to the PDF file
            source_type: Type of source (TEXTBOOK, PYQ, SYLLABUS)
            book_title: Optional title of the book
            chapter: Optional chapter number

        Returns:
            List of Document objects (one per page or chapter)
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return []

        doc_name = path.stem
        title = book_title or doc_name.replace("_", " ").title()
        documents: List[Document] = []

        logger.info(f"Loading PDF: {path.name} (Source: {source_type.value})")

        # Attempt extraction using PyMuPDF (fastest and most accurate)
        extracted_pages = self._extract_with_fitz(path)
        if not extracted_pages:
            # Fallback to pdfplumber
            logger.warning(f"PyMuPDF yielded no text for {path.name}. Trying pdfplumber fallback...")
            extracted_pages = self._extract_with_pdfplumber(path)
        if not extracted_pages:
            # Fallback to pypdf
            logger.warning(f"pdfplumber yielded no text. Trying pypdf fallback...")
            extracted_pages = self._extract_with_pypdf(path)

        for page_num, raw_text in enumerate(extracted_pages, start=1):
            if not raw_text or not raw_text.strip():
                continue

            cleaned_text = self.cleaner.clean_text(raw_text) if self.use_cleaner else raw_text
            if not cleaned_text or len(cleaned_text) < 30:
                continue

            unit, topic = self.tagger.tag_unit_and_topic(cleaned_text)
            chunk_id = f"{doc_name}_p{page_num}"

            doc = Document(
                chunk_id=chunk_id,
                text=cleaned_text,
                source_type=source_type,
                book=title if source_type == SourceType.TEXTBOOK else None,
                chapter=chapter,
                page_start=page_num,
                page_end=page_num,
                unit=unit,
                topic=topic,
                char_count=len(cleaned_text),
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} pages from {path.name}")
        return documents

    def _extract_with_fitz(self, path: Path) -> List[str]:
        """Extracts text using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            pages = [page.get_text("text") for page in doc]
            doc.close()
            return pages
        except ImportError:
            logger.debug("PyMuPDF (fitz) not installed, skipping.")
            return []
        except Exception as e:
            logger.warning(f"PyMuPDF error reading {path.name}: {e}")
            return []

    def _extract_with_pdfplumber(self, path: Path) -> List[str]:
        """Extracts text using pdfplumber."""
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages.append(text)
            return pages
        except ImportError:
            logger.debug("pdfplumber not installed, skipping.")
            return []
        except Exception as e:
            logger.warning(f"pdfplumber error reading {path.name}: {e}")
            return []

    def _extract_with_pypdf(self, path: Path) -> List[str]:
        """Extracts text using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return pages
        except Exception as e:
            logger.error(f"pypdf error reading {path.name}: {e}")
            return []

    def load_pyq_json(self, json_path: Union[str, Path]) -> List[Document]:
        """
        Loads structured PYQ question papers from JSON dataset into Document objects.
        Every question becomes its own atomic Document with full metadata.
        """
        path = Path(json_path)
        if not path.exists():
            logger.error(f"PYQ JSON not found at {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            papers = json.load(f)

        documents: List[Document] = []
        for paper in papers:
            paper_id = paper.get("paper_id", "RGPV_PYQ")
            year = paper.get("year")
            semester = paper.get("semester", 4)
            subject = paper.get("subject", "Operating Systems")

            for q in paper.get("questions", []):
                q_no = q.get("question_no", "")
                q_text = q.get("question_text", "")
                q_hindi = q.get("question_text_hindi", "")
                marks = q.get("marks", 7)
                unit = q.get("unit")
                topic = q.get("topic")

                # If unit/topic missing, auto-tag
                if not unit or not topic:
                    auto_unit, auto_topic = self.tagger.tag_unit_and_topic(q_text)
                    unit = unit or auto_unit
                    topic = topic or auto_topic

                cleaned_q = self.cleaner.clean_pyq_question(q_text)
                full_text = f"Question {q_no} ({marks} Marks) [RGPV {year}]:\n{cleaned_q}"
                if q_hindi:
                    full_text += f"\n(Hindi: {q_hindi})"

                chunk_id = f"{paper_id}_q{q_no.replace('.', '_')}"

                doc = Document(
                    chunk_id=chunk_id,
                    text=full_text,
                    source_type=SourceType.PYQ,
                    subject=subject,
                    year=year,
                    semester=semester,
                    marks=marks,
                    question_no=q_no,
                    unit=unit,
                    topic=topic,
                    char_count=len(full_text),
                )
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} atomic PYQ question documents from {path.name}")
        return documents

    def load_syllabus_file(self, syllabus_path: Union[str, Path]) -> List[Document]:
        """
        Loads the official syllabus file and splits it unit by unit.
        Each syllabus unit becomes an atomic Document.
        """
        path = Path(syllabus_path)
        if not path.exists():
            logger.error(f"Syllabus file not found at {path}")
            return []

        # If it's a JSON syllabus
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            docs = []
            for u in data.get("units", []):
                u_num = u.get("unit_number")
                title = u.get("title", "")
                topics = u.get("topics", [])
                text = f"RGPV OS Syllabus — Unit {u_num}: {title}\n" + "\n".join(f"- {t}" for t in topics)
                doc = Document(
                    chunk_id=f"rgpv_os_syllabus_unit_{u_num}",
                    text=text,
                    source_type=SourceType.SYLLABUS,
                    unit=u_num,
                    topic=title,
                    char_count=len(text),
                )
                docs.append(doc)
            return docs

        # Otherwise read text file and split on UNIT markers
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        import re
        unit_blocks = re.split(r"(?:={10,}|\bUNIT\s+\d+)", content)
        docs = []
        unit_num = 1
        for block in unit_blocks:
            cleaned = self.cleaner.clean_text(block)
            if cleaned and len(cleaned) > 50 and "RGPV" in cleaned or "Operating Systems" in cleaned or "UNIT" in block:
                unit, topic = self.tagger.tag_unit_and_topic(cleaned)
                docs.append(Document(
                    chunk_id=f"rgpv_os_syllabus_unit_{unit}",
                    text=cleaned,
                    source_type=SourceType.SYLLABUS,
                    unit=unit,
                    topic=topic,
                    char_count=len(cleaned),
                ))
        return docs
