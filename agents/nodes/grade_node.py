"""
Relevance grading node.
Filters retrieved documents into relevant and irrelevant subsets using NLI verification.
"""

from __future__ import annotations

from typing import Optional

from agents.state import GraphState
from retrieval.nli_grader import NLIGrader
from utils.logger import get_logger

logger = get_logger(__name__)

_grader: Optional[NLIGrader] = None

# Minimum relevant docs needed to proceed without retry
MIN_RELEVANT_DOCS = 2


def get_grader() -> NLIGrader:
    global _grader
    if _grader is None:
        _grader = NLIGrader()
    return _grader


def grade_node(state: GraphState) -> GraphState:
    """
    Grades candidate documents for relevance to the query using NLI.
    Sets state['relevant_docs'].

    Router decision (in graph.py) based on output:
      - len(relevant_docs) >= MIN_RELEVANT_DOCS → 'pass'
      - retrieval_attempts < 2                  → 'retry'
      - retrieval_attempts >= 2                 → 'web_search'
    """
    query = state.get("rewritten_query") or state["query"]
    candidates = state.get("candidates", [])

    if not candidates:
        logger.warning("[grade_node] No candidates to grade")
        return {**state, "relevant_docs": []}

    relevant, irrelevant = get_grader().grade_documents(
        query=query,
        documents=candidates,
    )

    logger.info(
        "[grade_node] %d candidates → %d relevant, %d irrelevant",
        len(candidates), len(relevant), len(irrelevant),
    )

    return {**state, "relevant_docs": relevant}


def route_after_grade(state: GraphState) -> str:
    """
    Conditional edge: decides next node after grading.
    Returns one of: 'pass' | 'retry' | 'web_search'
    """
    relevant = state.get("relevant_docs", [])
    attempts = state.get("retrieval_attempts", 0)

    if len(relevant) >= MIN_RELEVANT_DOCS:
        logger.info("[route_after_grade] PASS — %d relevant docs", len(relevant))
        return "pass"

    if attempts >= 2:
        logger.info("[route_after_grade] WEB_SEARCH — KB exhausted after %d attempts", attempts)
        return "web_search"

    logger.info("[route_after_grade] RETRY — only %d relevant docs, attempt %d", len(relevant), attempts)
    return "retry"
