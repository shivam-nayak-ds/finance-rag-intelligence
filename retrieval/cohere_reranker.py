"""Cohere Rerank v3 integration for result reranking."""

import os
import time
from typing import List, Optional, Tuple

from config.settings import RERANK_TOP_N
from models.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
RERANK_MODEL = "rerank-english-v3.0"


class CohereReranker:
    """
    Reranks retrieved documents using Cohere Rerank v3.

    Falls back to dense_score / rrf_score ordering if Cohere is unavailable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = RERANK_MODEL,
        top_n: int = RERANK_TOP_N,
    ) -> None:
        self.model = model
        self.top_n = top_n
        self._client = None

        key = api_key or COHERE_API_KEY
        if key:
            try:
                import cohere
                self._client = cohere.Client(api_key=key)
                logger.info("Cohere reranker initialised (model=%s)", model)
            except ImportError:
                logger.warning("cohere package not installed — pip install cohere")
            except Exception as err:
                logger.warning("Cohere client init failed: %s", err)
        else:
            logger.warning("COHERE_API_KEY not set — reranker will use fallback ordering")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents for a query.

        Args:
            query: The user query.
            documents: Candidate documents from hybrid retrieval.
            top_n: Number of top documents to return.

        Returns:
            List of (Document, reranker_score) tuples, sorted descending by score.
        """
        n = top_n or self.top_n

        if not documents:
            return []

        if self._client is None:
            return self._fallback_rank(documents, n)

        texts = [doc.text[:2048] for doc in documents]  # Cohere max input
        try:
            t0 = time.perf_counter()
            response = self._client.rerank(
                query=query,
                documents=texts,
                model=self.model,
                top_n=n,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("Cohere rerank: %d → %d results in %.1f ms", len(documents), n, elapsed_ms)

            ranked: List[Tuple[Document, float]] = []
            for result in response.results:
                doc = documents[result.index].model_copy()
                score = float(result.relevance_score)
                doc.reranker_score = score
                ranked.append((doc, score))

            return ranked

        except Exception as err:
            logger.warning("Cohere rerank API error: %s — using fallback", err)
            return self._fallback_rank(documents, n)

    def _fallback_rank(
        self,
        documents: List[Document],
        top_n: int,
    ) -> List[Tuple[Document, float]]:
        """Fallback: rank by rrf_score → dense_score → 0."""
        ranked = sorted(
            documents,
            key=lambda d: d.rrf_score or d.dense_score or 0.0,
            reverse=True,
        )
        result = []
        for doc in ranked[:top_n]:
            score = doc.rrf_score or doc.dense_score or 0.0
            doc_copy = doc.model_copy()
            doc_copy.reranker_score = score
            result.append((doc_copy, score))
        logger.info("Fallback rerank: returning top %d by rrf/dense score", len(result))
        return result
