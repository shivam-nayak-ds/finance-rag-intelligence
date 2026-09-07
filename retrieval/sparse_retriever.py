"""BM25 sparse retriever for keyword-based matching."""

import json
import pickle
from pathlib import Path
from typing import List, Optional

from config.settings import BM25_INDEX_PATH, BM25_TOP_K, CHUNKS_JSON
from models.documents import Document, Intent
from utils.logger import get_logger

logger = get_logger(__name__)


class SparseRetriever:
    """BM25-based keyword retriever built from pre-processed chunks."""

    def __init__(
        self,
        chunks_path: Path = CHUNKS_JSON,
        index_path: Path = BM25_INDEX_PATH,
        top_k: int = BM25_TOP_K,
    ) -> None:
        self.top_k = top_k
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self._docs: List[Document] = []
        self._bm25 = None
        self._load_or_build_index()

    # ──────────────────────────────────────────
    # Index management
    # ──────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()

    def _build_index(self) -> None:
        """Build BM25 index from chunks.json and persist to disk."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank-bm25: pip install rank-bm25")

        if not self.chunks_path.exists():
            raise FileNotFoundError(f"chunks.json not found at {self.chunks_path}")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        self._docs = [Document.model_validate(c) for c in raw_chunks]
        corpus = [self._tokenize(doc.text) for doc in self._docs]
        self._bm25 = BM25Okapi(corpus)

        # Persist index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "docs": self._docs}, f)

        logger.info("BM25 index built with %d documents → saved to %s", len(self._docs), self.index_path)

    def _load_or_build_index(self) -> None:
        """Load cached BM25 index or build fresh."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                self._bm25 = data["bm25"]
                self._docs = data["docs"]
                logger.info("BM25 index loaded from cache (%d docs)", len(self._docs))
                return
            except Exception as err:
                logger.warning("BM25 cache load failed (%s), rebuilding...", err)

        self._build_index()

    # ──────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        intent: str = Intent.UNKNOWN,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Score all documents with BM25 and return top-k.

        Args:
            query: User query string.
            intent: Classified intent (used for source_type filtering).
            top_k: Override default top-k.

        Returns:
            List of Document objects with bm25_score populated.
        """
        if self._bm25 is None:
            logger.error("BM25 index not available")
            return []

        k = top_k or self.top_k
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair (score, doc) and sort descending
        scored_docs = sorted(
            zip(scores, self._docs),
            key=lambda x: x[0],
            reverse=True,
        )

        # Apply intent-based source filter
        intent_source_map = {
            Intent.PYQ_RETRIEVAL: "pyq",
            Intent.SYLLABUS_LOOKUP: "syllabus",
            Intent.CONCEPT_EXPLANATION: None,  # all sources
        }
        source_filter = intent_source_map.get(intent)

        results: List[Document] = []
        for score, doc in scored_docs:
            if score <= 0:
                break
            if source_filter and doc.source_type != source_filter:
                continue
            doc_copy = doc.model_copy()
            doc_copy.bm25_score = float(score)
            results.append(doc_copy)
            if len(results) >= k:
                break

        logger.info("BM25 retrieval: %d docs (query=%r)", len(results), query[:50])
        return results
