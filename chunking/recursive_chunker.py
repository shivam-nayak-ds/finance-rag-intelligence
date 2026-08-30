"""
Recursive Academic Chunker (Production-Grade)
==============================================
Splits continuous academic textbook text into semantically cohesive chunks.
Preserves page numbers, chapters, syllabus units, and topic metadata across
every generated chunk for precise citation attribution.
"""

from typing import Final, List, Optional, Protocol, Sequence, runtime_checkable

from config.settings import TEXTBOOK_CHUNK_OVERLAP, TEXTBOOK_CHUNK_SIZE
from models.documents import Document, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class BaseChunker(Protocol):
    """Protocol defining the chunking interface for loose coupling."""

    def split_document(self, document: Document) -> List[Document]:
        """Splits a single Document into one or more smaller Document chunks."""
        ...

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """Batch splits multiple Documents."""
        ...


class CSRecursiveChunker:
    """
    Production-grade recursive text chunker for Computer Science textbooks.
    Recursively attempts natural boundaries: paragraphs -> sentences -> words.
    """

    # Priority order of natural text separation boundaries
    _DEFAULT_SEPARATORS: Final[List[str]] = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentence ends
        "? ",
        "! ",
        "; ",    # Clauses
        ", ",
        " ",     # Words
    ]

    def __init__(
        self,
        chunk_size: int = TEXTBOOK_CHUNK_SIZE,
        chunk_overlap: int = TEXTBOOK_CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes recursive chunker with bounds validation.

        Args:
            chunk_size: Target maximum characters per chunk (default: 800).
            chunk_overlap: Overlap between consecutive chunks (default: 150).
            separators: Optional list of separator strings in priority order.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self._DEFAULT_SEPARATORS

    def split_document(self, document: Document) -> List[Document]:
        """
        Splits a single document into enriched chunks, propagating full metadata.

        Args:
            document: Parent Document (e.g. textbook page).

        Returns:
            List of child Document chunks with inherited and enriched metadata.
        """
        if not document or not document.text or not document.text.strip():
            return []

        text = document.text.strip()

        # If text is already within chunk size limit, return as single chunk
        if len(text) <= self.chunk_size:
            return [
                document.model_copy(
                    update={
                        "chunk_index": 0,
                        "char_count": len(text),
                    }
                )
            ]

        # Generate text pieces using recursive splitting
        text_chunks = self._split_text_recursive(text, self.separators)

        # Merge pieces with overlap
        merged_chunks = self._merge_splits(text_chunks)

        total_chunks = len(merged_chunks)
        child_documents: List[Document] = []

        for idx, chunk_text in enumerate(merged_chunks):
            chunk_id = f"{document.chunk_id}_c{idx}"

            child_doc = Document(
                chunk_id=chunk_id,
                text=chunk_text,
                source_type=document.source_type,
                subject=document.subject,
                university=document.university,
                book=document.book,
                chapter=document.chapter,
                page_start=document.page_start,
                page_end=document.page_end,
                unit=document.unit,
                topic=document.topic,
                year=document.year,
                semester=document.semester,
                marks=document.marks,
                question_no=document.question_no,
                chunk_index=idx,
                char_count=len(chunk_text),
            )
            child_documents.append(child_doc)

        return child_documents

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """
        Batch splits a sequence of documents.
        """
        all_chunks: List[Document] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using the highest-priority separator that exists in the text.
        """
        final_chunks: List[str] = []

        # Find first applicable separator
        separator = separators[-1]  # Default fallback: space or character
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split text on chosen separator
        splits = text.split(separator) if separator else list(text)

        # Process each split component
        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if not new_separators:
                    # Can't split further, accept as is
                    good_splits.append(s)
                else:
                    # Recurse with lower-priority separators
                    sub_splits = self._split_text_recursive(s, new_separators)
                    good_splits.extend(sub_splits)

        return good_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Combines small split components into target chunk_size chunks while adding chunk_overlap.
        """
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for split in splits:
            split_len = len(split)
            if total_len + split_len > self.chunk_size:
                if total_len > 0:
                    joined = " ".join(current_doc).strip()
                    if joined:
                        docs.append(joined)

                    # Backtrack to maintain overlap
                    while total_len > self.chunk_overlap and current_doc:
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + 1

                current_doc.append(split)
                total_len += split_len + 1
            else:
                current_doc.append(split)
                total_len += split_len + 1

        if current_doc:
            joined = " ".join(current_doc).strip()
            if joined:
                docs.append(joined)

        return docs
