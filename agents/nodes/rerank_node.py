"""Rerank node — Cohere reranking + confidence scoring."""

from __future__ import annotations

import time
from typing import Optional

from agents.state import GraphState
from retrieval.cohere_reranker import CohereReranker
from tools.confidence_scorer import ConfidenceScorer
from utils.logger import get_logger

logger = get_logger(__name__)

_reranker: Optional[CohereReranker] = None
_scorer: Optional[ConfidenceScorer] = None


def get_reranker() -> CohereReranker:
    global _reranker
    if _reranker is None:
        _reranker = CohereReranker()
    return _reranker


def get_scorer() -> ConfidenceScorer:
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer()
    return _scorer


def rerank_node(state: GraphState) -> GraphState:
    """
    Reranks relevant docs via Cohere Rerank v3 (or fallback RRF ordering).
    Computes confidence level and warning message.
    Sets state['ranked_docs'], state['confidence'], state['confidence_score'],
    state['warning'], state['reranking_time_ms'].
    """
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("relevant_docs", [])

    if not docs:
        logger.warning("[rerank_node] No relevant docs to rerank")
        return {
            **state,
            "ranked_docs": [],
            "confidence": "low",
            "confidence_score": 0.0,
            "warning": "⚠️ Koi relevant content nahi mila — textbook se verify karein.",
        }

    t0 = time.perf_counter()
    ranked_docs = get_reranker().rerank(query=query, documents=docs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    confidence_level, confidence_score = get_scorer().score(ranked_docs)
    warning = get_scorer().warning_message(confidence_level)

    logger.info(
        "[rerank_node] %d docs reranked in %.1f ms → confidence=%s (%.3f)",
        len(docs), elapsed_ms, confidence_level, confidence_score,
    )

    return {
        **state,
        "ranked_docs": ranked_docs,
        "confidence": confidence_level,
        "confidence_score": confidence_score,
        "warning": warning,
        "reranking_time_ms": state.get("reranking_time_ms", 0.0) + elapsed_ms,
    }
