"""Document loader supporting PDF, JSON, and text file formats with multi-engine fallback."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, List, Optional, Protocol, Sequence, Set, Union, runtime_checkable

from ingestion.data_cleaner import BaseCleaner, DataCleaner
from ingestion.exceptions import (
    CorruptedFileError,
    DocumentParsingError,
    UnsupportedFileFormatError,
)
from ingestion.metadata_tagger import BaseMetadataTagger, MetadataTagger
from models.documents import Document, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoaderStats:
    """Statistics tracked during document loading."""
    total_files_processed: int = 0
    total_pages_scanned: int = 0
    total_documents_created: int = 0
    total_skipped_items: int = 0
    errors: List[str] = field(default_factory=list)


@runtime_checkable
class BaseDocumentLoader(Protocol):
    """Protocol for document loader implementations."""

    def load_pdf(
        self,
        file_path: Union[str, Path],
        source_type: SourceType = SourceType.TEXTBOOK,
        book_title: Optional[str] = None,
        chapter: Optional[int] = None,
    ) -> List[Document]:
        """Loads PDF pages into Document objects."""
        ...

    def load_pyq_json(self, json_path: Union[str, Path]) -> List[Document]:
        """Loads PYQs from structured JSON."""
        ...

    def load_syllabus(self, syllabus_path: Union[str, Path]) -> List[Document]:
        """Loads syllabus units into Document objects."""
        ...


class PDFLoader:
    """Document loader supporting multi-engine PDF parsing, JSON datasets, and syllabus files."""

    _SUPPORTED_PDF_EXTENSIONS: Final[Set[str]] = {".pdf"}
    _SUPPORTED_DATA_EXTENSIONS: Final[Set[str]] = {".json", ".txt", ".md"}
    _MIN_PAGE_CHARACTERS: Final[int] = 30

    def __init__(
        self,
        cleaner: Optional[BaseCleaner] = None,
        tagger: Optional[BaseMetadataTagger] = None,
        clean_extracted_text: bool = True,
    ) -> None:
        """
        Initializes loader with optional dependency-injected cleaner and tagger.
        """
        self.cleaner: BaseCleaner = cleaner or DataCleaner()
        self.tagger: BaseMetadataTagger = tagger or MetadataTagger()
        self.clean_extracted_text = clean_extracted_text
        self.stats = LoaderStats()

    def load_pdf(
        self,
        file_path: Union[str, Path],
        source_type: SourceType = SourceType.TEXTBOOK,
        book_title: Optional[str] = None,
        chapter: Optional[int] = None,
    ) -> List[Document]:
        """
        Loads PDF document page-by-page into standardized Document objects.

        Args:
            file_path: Path to the target PDF file.
            source_type: Source categorization (TEXTBOOK, PYQ, SYLLABUS).
            book_title: Optional book title override.
            chapter: Optional chapter number.

        Returns:
            List of successfully extracted Document objects.

        Raises:
            FileNotFoundError: If file_path does not exist.
            UnsupportedFileFormatError: If file is not a PDF.
            CorruptedFileError: If all extraction engines fail to parse the file.
        """
        path = Path(file_path)
        self._validate_file_path(path, allowed_extensions=self._SUPPORTED_PDF_EXTENSIONS)

        doc_stem = path.stem
        title = book_title or doc_stem.replace("_", " ").title()
        documents: List[Document] = []

        logger.info(f"Opening PDF: '{path.name}' ({path.stat().st_size / (1024 * 1024):.2f} MB)")

        # Engine fallback sequence
        extracted_pages = self._extract_with_fitz(path)
        if not extracted_pages:
            logger.warning(f"PyMuPDF yielded 0 pages for '{path.name}'. Attempting fallback to pdfplumber...")
            extracted_pages = self._extract_with_pdfplumber(path)

        if not extracted_pages:
            logger.warning(f"pdfplumber yielded 0 pages. Attempting fallback to pypdf...")
            extracted_pages = self._extract_with_pypdf(path)

        if not extracted_pages:
            err_msg = f"Failed to extract any text from '{path.name}' with all available PDF engines."
            logger.error(err_msg)
            self.stats.errors.append(err_msg)
            raise CorruptedFileError(err_msg, file_path=str(path))

        self.stats.total_files_processed += 1
        self.stats.total_pages_scanned += len(extracted_pages)

        # Process each page with isolated error handling
        for page_num, raw_text in enumerate(extracted_pages, start=1):
            try:
                if not raw_text or not raw_text.strip():
                    self.stats.total_skipped_items += 1
                    continue

                cleaned_text = (
                    self.cleaner.clean(raw_text)
                    if self.clean_extracted_text
                    else raw_text.strip()
                )

                if len(cleaned_text) < self._MIN_PAGE_CHARACTERS:
                    self.stats.total_skipped_items += 1
                    continue

                unit, topic = self.tagger.tag_unit_and_topic(cleaned_text)
                chunk_id = f"{doc_stem}_p{page_num}"

                # Auto-detect chapter from text heading or page number
                detected_chapter = chapter or self.tagger.detect_chapter(
                    cleaned_text, page_number=page_num
                )

                doc = Document(
                    chunk_id=chunk_id,
                    text=cleaned_text,
                    source_type=source_type,
                    book=title if source_type == SourceType.TEXTBOOK else None,
                    chapter=detected_chapter,
                    page_start=page_num,
                    page_end=page_num,
                    unit=unit,
                    topic=topic,
                    char_count=len(cleaned_text),
                )
                documents.append(doc)
            except Exception as page_err:
                # Isolate page error so remaining pages are preserved
                logger.warning(f"Error parsing page {page_num} of '{path.name}': {page_err}")
                self.stats.errors.append(f"{path.name} p.{page_num}: {str(page_err)}")
                continue

        self.stats.total_documents_created += len(documents)
        logger.info(f"Successfully extracted {len(documents)} pages from '{path.name}'")
        return documents

    def _extract_with_fitz(self, path: Path) -> List[str]:
        """Extracts text using PyMuPDF (fitz) within a context manager for memory safety."""
        try:
            import fitz
            pages: List[str] = []
            with fitz.open(str(path)) as doc:
                for page in doc:
                    pages.append(page.get_text("text") or "")
            return pages
        except ImportError:
            logger.debug("PyMuPDF (fitz) not installed.")
            return []
        except Exception as err:
            logger.warning(f"PyMuPDF engine failed on '{path.name}': {err}")
            return []

    def _extract_with_pdfplumber(self, path: Path) -> List[str]:
        """Extracts text using pdfplumber within a context manager."""
        try:
            import pdfplumber
            pages: List[str] = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
            return pages
        except ImportError:
            logger.debug("pdfplumber not installed.")
            return []
        except Exception as err:
            logger.warning(f"pdfplumber engine failed on '{path.name}': {err}")
            return []

    def _extract_with_pypdf(self, path: Path) -> List[str]:
        """Extracts text using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return pages
        except Exception as err:
            logger.warning(f"pypdf engine failed on '{path.name}': {err}")
            return []

    def load_pyq_json(self, json_path: Union[str, Path]) -> List[Document]:
        """
        Loads atomic exam questions from structured PYQ JSON files.
        """
        path = Path(json_path)
        self._validate_file_path(path, allowed_extensions={".json"})

        try:
            with open(path, "r", encoding="utf-8") as f:
                papers = json.load(f)
        except json.JSONDecodeError as jde:
            raise DocumentParsingError(f"Malformed JSON in PYQ dataset: {jde}", file_path=str(path)) from jde

        if not isinstance(papers, list):
            raise DocumentParsingError("PYQ JSON root must be an array of exam papers.", file_path=str(path))

        documents: List[Document] = []
        for paper in papers:
            paper_id = paper.get("paper_id", "RGPV_PYQ")
            year = paper.get("year")
            semester = paper.get("semester", 4)
            subject = paper.get("subject", "Operating Systems")

            for q in paper.get("questions", []):
                q_no = q.get("question_no", "1")
                q_text = q.get("question_text", "")
                q_hindi = q.get("question_text_hindi", "")
                marks = q.get("marks", 7)
                unit = q.get("unit")
                topic = q.get("topic")

                if not q_text.strip():
                    self.stats.total_skipped_items += 1
                    continue

                if not unit or not topic:
                    auto_unit, auto_topic = self.tagger.tag_unit_and_topic(q_text)
                    unit = unit or auto_unit
                    topic = topic or auto_topic

                cleaned_q = self.cleaner.clean_pyq(q_text)
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

        self.stats.total_documents_created += len(documents)
        logger.info(f"Loaded {len(documents)} atomic PYQ question documents from '{path.name}'")
        return documents

    def load_syllabus(self, syllabus_path: Union[str, Path]) -> List[Document]:
        """
        Loads syllabus definitions and parses them into unit-by-unit documents.
        """
        path = Path(syllabus_path)
        self._validate_file_path(path, allowed_extensions=self._SUPPORTED_DATA_EXTENSIONS)

        docs: List[Document] = []

        if path.suffix == ".json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as jde:
                raise DocumentParsingError(f"Malformed JSON in syllabus file: {jde}", file_path=str(path)) from jde

            for u in data.get("units", []):
                u_num = u.get("unit_number", 1)
                title = u.get("title", "")
                topics = u.get("topics", [])
                text = (
                    f"RGPV OS Syllabus — Unit {u_num}: {title}\n"
                    + "\n".join(f"- {t}" for t in topics)
                )
                docs.append(
                    Document(
                        chunk_id=f"rgpv_os_syllabus_unit_{u_num}",
                        text=text,
                        source_type=SourceType.SYLLABUS,
                        unit=u_num,
                        topic=title,
                        char_count=len(text),
                    )
                )
            self.stats.total_documents_created += len(docs)
            logger.info(f"Loaded {len(docs)} syllabus unit documents from '{path.name}'")
            return docs

        # Plain text fallback
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        import re
        unit_blocks = re.split(r"(?:={10,}|\bUNIT\s+\d+)", content)
        for block in unit_blocks:
            cleaned = self.cleaner.clean(block)
            if cleaned and len(cleaned) > self._MIN_PAGE_CHARACTERS:
                unit, topic = self.tagger.tag_unit_and_topic(cleaned)
                docs.append(
                    Document(
                        chunk_id=f"rgpv_os_syllabus_unit_{unit}",
                        text=cleaned,
                        source_type=SourceType.SYLLABUS,
                        unit=unit,
                        topic=topic,
                        char_count=len(cleaned),
                    )
                )

        self.stats.total_documents_created += len(docs)
        logger.info(f"Loaded {len(docs)} syllabus unit documents from '{path.name}'")
        return docs

    @classmethod
    def _validate_file_path(cls, path: Path, allowed_extensions: Set[str]) -> None:
        """Validates file existence and extension constraints."""
        if not path.exists():
            raise FileNotFoundError(f"Target file does not exist: {path}")

        if not path.is_file():
            raise IsADirectoryError(f"Expected a file path but found a directory: {path}")

        if path.suffix.lower() not in allowed_extensions:
            raise UnsupportedFileFormatError(
                f"Unsupported file format '{path.suffix}'. Allowed: {', '.join(allowed_extensions)}",
                file_path=str(path),
            )
