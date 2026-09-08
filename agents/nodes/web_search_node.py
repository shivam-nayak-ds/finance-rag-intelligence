"""
Web search node — Tavily fallback when KB retrieval is exhausted.
Converts web results into Document objects with SourceType.WEB.
"""

from __future__ import annotations

from typing import Optional

from agents.state import GraphState
from tools.web_search import WebSearchTool
from utils.logger import get_logger

logger = get_logger(__name__)

_web_searcher: Optional[WebSearchTool] = None


def get_web_searcher() -> WebSearchTool:
    global _web_searcher
    if _web_searcher is None:
        _web_searcher = WebSearchTool()
    return _web_searcher


def web_search_node(state: GraphState) -> GraphState:
    """
    Triggers Tavily web search when the knowledge base has insufficient
    relevant content after max retrieval retries.

    Merges web results into state['candidates'] and state['relevant_docs'].
    Sets state['web_search_used'] = True.
    """
    query = state.get("rewritten_query") or state["query"]

    logger.info("[web_search_node] KB exhausted — searching web for: %r", query[:80])

    web_docs = get_web_searcher().search(query=query, max_results=3)

    if not web_docs:
        logger.warning("[web_search_node] Web search returned 0 results")
        return {**state, "web_docs": [], "web_search_used": True}

    logger.info("[web_search_node] %d web results retrieved", len(web_docs))

    # Merge web docs into relevant_docs (web search = pre-graded relevant)
    existing_relevant = state.get("relevant_docs", [])
    merged_relevant = existing_relevant + web_docs

    return {
        **state,
        "web_docs": web_docs,
        "relevant_docs": merged_relevant,
        "candidates": state.get("candidates", []) + web_docs,
        "web_search_used": True,
    }
