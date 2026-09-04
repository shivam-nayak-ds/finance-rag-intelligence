"""Syllabus chunking preserving unit boundaries."""

from typing import List, Sequence

from models.documents import Document, SourceType
from utils.logger import get_logger

logger = get_logger(__name__)


class SyllabusChunker:
    """Processes syllabus documents by preserving unit boundaries."""

    def split_document(self, document: Document) -> List[Document]:
        """Processes a single syllabus document without splitting unit contents."""
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
        """Batch processes syllabus documents."""
        syllabus_chunks: List[Document] = []
        for doc in documents:
            if doc.source_type == SourceType.SYLLABUS:
                syllabus_chunks.extend(self.split_document(doc))
        return syllabus_chunks
