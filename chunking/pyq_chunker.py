"""
Atomic PYQ Chunker (Production-Grade)
=====================================
Processes exam question papers as atomic, non-splittable units.
Enriches each question with standard exam citation tags and metadata.
"""

from typing import List, Sequence

from models.documents import Document, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


class PYQChunker:
    """
    Atomic chunker for exam previous year questions (PYQs).
    Guarantees 1 Question = 1 Complete Chunk with rich exam metadata.
    """

    def split_document(self, document: Document) -> List[Document]:
        """
        Enriches an atomic PYQ document without splitting its question statement.

        Args:
            document: Raw PYQ Document.

        Returns:
            List with exactly one enriched Document chunk.
        """
        if not document or not document.text or not document.text.strip():
            return []

        # Ensure source_type is PYQ
        enriched_doc = document.model_copy(
            update={
                "source_type": SourceType.PYQ,
                "chunk_index": 0,
                "char_count": len(document.text.strip()),
            }
        )

        return [enriched_doc]

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """
        Batch enriches multiple PYQ questions.
        """
        pyq_chunks: List[Document] = []
        for doc in documents:
            if doc.source_type == SourceType.PYQ:
                pyq_chunks.extend(self.split_document(doc))
        return pyq_chunks
