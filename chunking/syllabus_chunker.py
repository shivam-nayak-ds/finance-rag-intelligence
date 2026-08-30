"""
Syllabus Unit Chunker (Production-Grade)
========================================
Maintains unit-boundary integrity for university syllabus documents.
Guarantees that each syllabus unit remains a single, comprehensive chunk.
"""

from typing import List, Sequence

from models.documents import Document, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


class SyllabusChunker:
    """
    Chunker for syllabus documents maintaining unit boundaries.
    1 Syllabus Unit = 1 Chunk.
    """

    def split_document(self, document: Document) -> List[Document]:
        """
        Enriches a syllabus unit document as an atomic chunk.

        Args:
            document: Raw syllabus unit document.

        Returns:
            List with exactly one enriched Document chunk.
        """
        if not document or not document.text or not document.text.strip():
            return []

        enriched_doc = document.model_copy(
            update={
                "source_type": SourceType.SYLLABUS,
                "chunk_index": 0,
                "char_count": len(document.text.strip()),
            }
        )

        return [enriched_doc]

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """
        Batch enriches multiple syllabus documents.
        """
        syllabus_chunks: List[Document] = []
        for doc in documents:
            if doc.source_type == SourceType.SYLLABUS:
                syllabus_chunks.extend(self.split_document(doc))
        return syllabus_chunks
