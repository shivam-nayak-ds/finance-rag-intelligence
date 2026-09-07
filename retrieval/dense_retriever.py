"""Dense vector retriever backed by Qdrant."""

from typing import List, Optional

from config.settings import (
    DENSE_TOP_K,
    QDRANT_COLLECTION_PYQS,
    QDRANT_COLLECTION_SYLLABUS,
    QDRANT_COLLECTION_TEXTBOOK,
)
from embedding.hf_embedder import HFEmbedder
from models.documents import Document, Intent
from utils.logger import get_logger
from vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


class DenseRetriever:
    """Performs semantic search across Qdrant collections using dense embeddings."""

    def __init__(
        self,
        embedder: Optional[HFEmbedder] = None,
        store: Optional[QdrantStore] = None,
        top_k: int = DENSE_TOP_K,
    ) -> None:
        self.embedder = embedder or HFEmbedder()
        self.store = store or QdrantStore()
        self.top_k = top_k

    def _collections_for_intent(self, intent: str) -> List[str]:
        """Map query intent to the most relevant Qdrant collections."""
        mapping = {
            Intent.PYQ_RETRIEVAL: [QDRANT_COLLECTION_PYQS],
            Intent.SYLLABUS_LOOKUP: [QDRANT_COLLECTION_SYLLABUS],
            Intent.CONCEPT_EXPLANATION: [QDRANT_COLLECTION_TEXTBOOK, QDRANT_COLLECTION_SYLLABUS],
            Intent.TOPIC_IMPORTANCE: [QDRANT_COLLECTION_PYQS, QDRANT_COLLECTION_SYLLABUS],
        }
        return mapping.get(intent, [QDRANT_COLLECTION_TEXTBOOK, QDRANT_COLLECTION_PYQS, QDRANT_COLLECTION_SYLLABUS])

    def retrieve(
        self,
        query: str,
        intent: str = Intent.UNKNOWN,
        top_k: Optional[int] = None,
        unit: Optional[int] = None,
        topic: Optional[str] = None,
    ) -> List[Document]:
        """
        Embed the query and retrieve top-k semantically similar chunks.

        Args:
            query: User's (possibly rewritten) query string.
            intent: Classified intent to select collections.
            top_k: Override default top-k.
            unit: Optional syllabus unit filter (1-5).
            topic: Optional topic keyword filter.

        Returns:
            List of Document objects with dense_score populated.
        """
        k = top_k or self.top_k
        query_vector = self.embedder.embed_query(query)
        collections = self._collections_for_intent(intent)

        results: List[Document] = []
        for collection in collections:
            try:
                docs = self.store.similarity_search(
                    collection_name=collection,
                    query_vector=query_vector,
                    top_k=k,
                    unit=unit,
                    topic=topic,
                )
                results.extend(docs)
                logger.debug("Dense[%s]: retrieved %d docs", collection, len(docs))
            except Exception as err:
                logger.warning("Dense retrieval failed for collection '%s': %s", collection, err)

        # Sort by dense_score descending, deduplicate by chunk_id
        seen: set[str] = set()
        unique_results: List[Document] = []
        for doc in sorted(results, key=lambda d: d.dense_score or 0.0, reverse=True):
            if doc.chunk_id not in seen:
                seen.add(doc.chunk_id)
                unique_results.append(doc)

        logger.info("Dense retrieval: %d unique docs from %s", len(unique_results), collections)
        return unique_results[:k]
