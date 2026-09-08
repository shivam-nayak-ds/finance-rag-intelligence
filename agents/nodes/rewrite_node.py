"""Query rewrite node — expands and improves the query for better retrieval."""

from __future__ import annotations

from typing import Optional

from agents.state import GraphState
from generation.memory import ConversationMemory, get_memory
from tools.query_rewriter import QueryRewriter
from utils.logger import get_logger

logger = get_logger(__name__)

_rewriter: Optional[QueryRewriter] = None
_memory: Optional[ConversationMemory] = None


def get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


def get_mem() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = get_memory()
    return _memory


def rewrite_node(state: GraphState) -> GraphState:
    """
    Rewrites the query using LLM-based expansion.
    On retry (retrieval_attempts > 0), adds a "rephrase" hint.
    Sets state['rewritten_query'] and state['context_hint'].
    """
    query = state["query"]
    session_id = state.get("session_id")
    attempts = state.get("retrieval_attempts", 0)

    # Build context hint from short-term memory
    context_hint = ""
    if session_id:
        context_hint = get_mem().get_context_summary(session_id) or ""

    # On retry: add explicit instruction to rephrase differently
    if attempts > 0:
        query_for_rewrite = (
            f"[Previous retrieval failed. Rephrase differently.] {query}"
        )
        logger.info("[rewrite_node] retry attempt=%d — adding rephrase hint", attempts)
    else:
        query_for_rewrite = query

    rewritten = get_rewriter().rewrite(query_for_rewrite, context=context_hint)
    logger.info("[rewrite_node] %r → %r", query[:60], rewritten[:80])

    return {**state, "rewritten_query": rewritten, "context_hint": context_hint}
