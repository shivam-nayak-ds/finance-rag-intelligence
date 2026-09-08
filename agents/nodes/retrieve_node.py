"""Retrieve node — runs hybrid dense+sparse retrieval and tracks attempt count."""

from __future__ import annotations

import time
from typing import Optional

from agents.state import GraphState
from retrieval.hybrid_retriever import HybridRetriever
from utils.logger import get_logger

logger = get_logger(__name__)

_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve_node(state: GraphState) -> GraphState:
    """
    Runs hybrid retrieval (Dense + BM25 + RRF) on the rewritten query.
    Increments retrieval_attempts counter.
    Sets state['candidates'] and state['retrieval_time_ms'].
    """
    query = state.get("rewritten_query") or state["query"]
    intent = state.get("intent", "unknown")
    unit = state.get("unit")
    attempts = state.get("retrieval_attempts", 0)

    t0 = time.perf_counter()
    candidates = get_retriever().retrieve(query=query, intent=intent, unit=unit)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "[retrieve_node] attempt=%d → %d candidates in %.1f ms",
        attempts + 1, len(candidates), elapsed_ms,
    )

    return {
        **state,
        "candidates": candidates,
        "retrieval_attempts": attempts + 1,
        "retrieval_time_ms": state.get("retrieval_time_ms", 0.0) + elapsed_ms,
    }
