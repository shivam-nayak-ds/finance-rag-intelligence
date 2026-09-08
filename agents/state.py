"""State definition for SyllAIq execution graph."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple, TypedDict

from models.documents import Citation, ConfidenceLevel, Document, Intent
from models.responses import RAGResult


class GraphState(TypedDict, total=False):
    """
    Shared state object for the SyllAIq LangGraph pipeline.

    Fields are optional (total=False) so nodes only need to set
    what they produce — LangGraph merges state across nodes.
    """

    # ── Input ────────────────────────────────────────────────
    query: str                          # original student query
    session_id: Optional[str]          # conversation memory key
    unit: Optional[int]                # syllabus unit filter (1-5)

    # ── Intent & Rewriting ───────────────────────────────────
    intent: str                         # classified Intent enum value
    rewritten_query: str               # LLM-expanded query for retrieval
    context_hint: str                  # conversation context for rewriter

    # ── Retrieval ────────────────────────────────────────────
    candidates: List[Document]         # raw hybrid-retrieved docs
    relevant_docs: List[Document]      # NLI-graded relevant subset
    ranked_docs: List[Tuple[Document, float]]  # Cohere-reranked (doc, score) pairs
    web_docs: List[Document]           # Tavily web search results

    # ── Confidence ───────────────────────────────────────────
    confidence: str                    # ConfidenceLevel enum value
    confidence_score: float            # raw 0-1 float
    warning: Optional[str]            # user-facing warning for low confidence

    # ── Generation ───────────────────────────────────────────
    answer: str                        # LLM-generated answer text
    citations: List[Citation]          # source citations list
    total_tokens: int                  # token usage

    # ── Control Flow (Self-RAG loops) ────────────────────────
    retrieval_attempts: int            # how many times we've retrieved (max 2)
    generation_attempts: int           # how many times we've generated (max 1)
    is_grounded: bool                  # NLI groundedness check result
    retrieval_failed: bool             # True if no good docs found after all retries
    web_search_used: bool              # True if Tavily was triggered

    # ── Timing ───────────────────────────────────────────────
    start_time: float                  # pipeline start timestamp
    retrieval_time_ms: float
    reranking_time_ms: float
    generation_time_ms: float

    # ── Final Output ─────────────────────────────────────────
    result: Optional[RAGResult]        # assembled final result
    error: Optional[str]               # error message if pipeline fails


def initial_state(
    query: str,
    session_id: Optional[str] = None,
    unit: Optional[int] = None,
) -> GraphState:
    """
    Build a fresh GraphState for a new query.
    Sets safe defaults for all control-flow counters.
    """
    return GraphState(
        query=query,
        session_id=session_id,
        unit=unit,
        intent=Intent.UNKNOWN,
        rewritten_query=query,
        context_hint="",
        candidates=[],
        relevant_docs=[],
        ranked_docs=[],
        web_docs=[],
        confidence=ConfidenceLevel.LOW,
        confidence_score=0.0,
        warning=None,
        answer="",
        citations=[],
        total_tokens=0,
        retrieval_attempts=0,
        generation_attempts=0,
        is_grounded=False,
        retrieval_failed=False,
        web_search_used=False,
        start_time=time.perf_counter(),
        retrieval_time_ms=0.0,
        reranking_time_ms=0.0,
        generation_time_ms=0.0,
        result=None,
        error=None,
    )
