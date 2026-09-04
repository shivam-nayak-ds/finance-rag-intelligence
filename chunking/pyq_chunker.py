"""Exam question chunking preserving atomic question boundaries."""

from typing import List, Sequence

from models.documents import Document, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


class PYQChunker:
    """Processes exam questions as atomic units without splitting question statements."""

    def split_document(self, document: Document) -> List[Document]:
        """Validates and prepares an atomic PYQ document."""
        if not document or not document.text or not document.text.strip():
            return []

        enriched_doc = document.model_copy(
            update={
                "source_type": SourceType.PYQ,
                "chunk_index": 0,
                "char_count": len(document.text.strip()),
            }
        )
        return [enriched_doc]

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """Batch processes PYQ documents."""
        pyq_chunks: List[Document] = []
        for doc in documents:
            if doc.source_type == SourceType.PYQ:
                pyq_chunks.extend(self.split_document(doc))
        return pyq_chunks
