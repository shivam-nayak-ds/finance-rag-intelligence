"""Hybrid retriever combining dense and sparse results via Reciprocal Rank Fusion."""

from typing import Dict, List, Optional

from config.settings import DENSE_TOP_K, BM25_TOP_K, RERANK_TOP_N, RRF_K
from models.documents import Document, Intent
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:
    """
    Merge multiple ranked lists using RRF.
    RRF score = sum(1 / (k + rank)) for each list the doc appears in.

    Args:
        ranked_lists: Each sub-list is a ranked result set.
        k: RRF constant (default 60).

    Returns:
        Merged list sorted by descending RRF score.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            cid = doc.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in doc_map:
                doc_map[cid] = doc

    merged: List[Document] = []
    for cid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        doc = doc_map[cid].model_copy()
        doc.rrf_score = rrf_score
        merged.append(doc)

    return merged


class HybridRetriever:
    """
    Combines dense (Qdrant) and sparse (BM25) retrievers via RRF fusion.

    Flow:
        query → [DenseRetriever, SparseRetriever] → RRF → top-N chunks
    """

    def __init__(
        self,
        dense: Optional[DenseRetriever] = None,
        sparse: Optional[SparseRetriever] = None,
        dense_top_k: int = DENSE_TOP_K,
        bm25_top_k: int = BM25_TOP_K,
        final_top_n: int = RERANK_TOP_N * 3,  # send more to reranker
        rrf_k: int = RRF_K,
    ) -> None:
        self.dense = dense or DenseRetriever(top_k=dense_top_k)
        self.sparse = sparse or SparseRetriever(top_k=bm25_top_k)
        self.final_top_n = final_top_n
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        intent: str = Intent.UNKNOWN,
        top_n: Optional[int] = None,
        unit: Optional[int] = None,
        topic: Optional[str] = None,
    ) -> List[Document]:
        """
        Run both retrievers in parallel, fuse with RRF, return top-n chunks.

        Args:
            query: Rewritten or original user query.
            intent: Classified intent for collection selection.
            top_n: Number of results to return (default: final_top_n).
            unit: Syllabus unit filter.
            topic: Topic keyword filter.

        Returns:
            Fused list of Document objects with rrf_score populated.
        """
        n = top_n or self.final_top_n

        dense_results = self.dense.retrieve(query, intent=intent, unit=unit, topic=topic)
        sparse_results = self.sparse.retrieve(query, intent=intent)

        if not dense_results and not sparse_results:
            logger.warning("Both dense and sparse returned empty results for query: %r", query[:60])
            return []

        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k=self.rrf_k,
        )

        logger.info(
            "Hybrid retrieval: dense=%d, sparse=%d → fused=%d → returning top %d",
            len(dense_results), len(sparse_results), len(fused), min(n, len(fused)),
        )
        return fused[:n]
