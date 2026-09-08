"""
Generate node — calls LLMChain to produce the answer.
On retry (generation_attempts > 0), uses a different subset of ranked docs
to avoid repeating the same hallucinated content.
"""

from __future__ import annotations

import time
from typing import Optional

from agents.state import GraphState
from generation.llm_chain import LLMChain
from generation.memory import ConversationMemory, LongTermMemory, get_memory, get_long_term_memory
from utils.logger import get_logger

logger = get_logger(__name__)

_llm_chain: Optional[LLMChain] = None
_memory: Optional[ConversationMemory] = None
_long_term: Optional[LongTermMemory] = None


def get_llm_chain() -> LLMChain:
    global _llm_chain
    if _llm_chain is None:
        _llm_chain = LLMChain()
    return _llm_chain


def get_mem() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = get_memory()
    return _memory


def get_ltm() -> LongTermMemory:
    global _long_term
    if _long_term is None:
        _long_term = get_long_term_memory()
    return _long_term


def generate_node(state: GraphState) -> GraphState:
    """
    Generates an answer using LLM with ranked docs as context.

    On first attempt: uses top-N ranked docs.
    On retry:         shifts doc window to avoid same hallucinated content.

    Sets state['answer'], state['citations'], state['total_tokens'],
    state['generation_attempts'], state['generation_time_ms'].
    """
    query = state["query"]
    session_id = state.get("session_id")
    intent = state.get("intent", "unknown")
    ranked_docs = state.get("ranked_docs", [])
    generation_attempts = state.get("generation_attempts", 0)
    confidence = state.get("confidence", "low")
    confidence_score = state.get("confidence_score", 0.0)
    warning = state.get("warning")

    # On retry: shift doc window (skip top-1, use next set)
    if generation_attempts > 0 and len(ranked_docs) > 1:
        doc_window = ranked_docs[1:]
        logger.info("[generate_node] retry=%d — shifting doc window", generation_attempts)
    else:
        doc_window = ranked_docs

    top_docs = [doc for doc, _ in doc_window]

    # Pull conversation history (excluding current user turn)
    history_msgs = []
    if session_id:
        all_msgs = get_mem().get_history_messages(session_id)
        history_msgs = all_msgs[:-1] if len(all_msgs) > 1 else []

    personalization = (
        get_ltm().get_personalization_hint(session_id) if session_id else None
    )

    t0 = time.perf_counter()
    result = get_llm_chain().generate(
        query=query,
        documents=top_docs,
        ranked_docs=doc_window,
        intent=intent,
        confidence=confidence,
        confidence_score=confidence_score,
        warning=warning,
        retrieval_time_ms=state.get("retrieval_time_ms", 0.0),
        reranking_time_ms=state.get("reranking_time_ms", 0.0),
        history_messages=history_msgs,
        personalization_hint=personalization,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "[generate_node] attempt=%d — generated %d chars in %.1f ms",
        generation_attempts + 1, len(result.answer), elapsed_ms,
    )

    return {
        **state,
        "answer": result.answer,
        "citations": result.citations,
        "total_tokens": result.total_tokens,
        "generation_attempts": generation_attempts + 1,
        "generation_time_ms": state.get("generation_time_ms", 0.0) + elapsed_ms,
    }
