"""
Finalize node — packages final GraphState into RAGResult and saves session memory.
"""

from __future__ import annotations

import time
from typing import Optional

from agents.state import GraphState
from generation.memory import ConversationMemory, LongTermMemory, get_memory, get_long_term_memory
from models.documents import ConfidenceLevel, Intent
from models.responses import RAGResult
from utils.logger import get_logger

logger = get_logger(__name__)

_memory: Optional[ConversationMemory] = None
_long_term: Optional[LongTermMemory] = None


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


def finalize_node(state: GraphState) -> GraphState:
    """
    Constructs the final RAGResult object, persists turn in conversation memory,
    and returns updated state.
    """
    # If state already has a prebuilt result (e.g. from SQL node), use it
    if state.get("result") is not None:
        result = state["result"]
        session_id = state.get("session_id")
        if session_id and result.answer:
            get_mem().add_turn(
                session_id=session_id,
                user_message=state["query"],
                assistant_message=result.answer,
                citations=[c.chunk_id for c in result.citations],
            )
            get_ltm().log_query(
                session_id=session_id,
                query=state["query"],
                intent=str(state.get("intent", "analytics")),
                unit=state.get("unit"),
            )
        return state

    total_latency_ms = (time.perf_counter() - state.get("start_time", time.perf_counter())) * 1000

    raw_intent = state.get("intent", Intent.UNKNOWN)
    try:
        intent_enum = Intent(raw_intent)
    except Exception:
        intent_enum = Intent.UNKNOWN

    raw_conf = state.get("confidence", ConfidenceLevel.LOW)
    try:
        conf_enum = ConfidenceLevel(raw_conf)
    except Exception:
        conf_enum = ConfidenceLevel.LOW

    warning = state.get("warning")
    is_grounded = state.get("is_grounded", False)
    gen_attempts = state.get("generation_attempts", 0)

    # If verification failed and no warning exists, append warning
    if not is_grounded and not warning:
        warning = "⚠️ Note: Groundedness check could not fully verify this answer. Please cross-check with textbook."

    # Did we self-correct?
    retrieval_attempts = state.get("retrieval_attempts", 0)
    self_corrected = (retrieval_attempts > 1) or (gen_attempts > 1)
    self_correction_count = max(0, (retrieval_attempts - 1)) + max(0, (gen_attempts - 1))

    result = RAGResult(
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        confidence=conf_enum,
        grounding_score=1.0 if is_grounded else 0.0,
        intent=intent_enum,
        self_corrected=self_corrected,
        self_correction_count=self_correction_count,
        retrieval_failed=state.get("retrieval_failed", False),
        web_search_used=state.get("web_search_used", False),
        retrieval_time_ms=state.get("retrieval_time_ms", 0.0),
        reranking_time_ms=state.get("reranking_time_ms", 0.0),
        generation_time_ms=state.get("generation_time_ms", 0.0),
        total_latency_ms=total_latency_ms,
        total_tokens=state.get("total_tokens", 0),
        warning=warning,
        error=state.get("error"),
        status="failed" if state.get("error") else ("partial" if warning else "success"),
    )

    # Persist in conversation memory
    session_id = state.get("session_id")
    if session_id and result.answer:
        get_mem().add_turn(
            session_id=session_id,
            user_message=state["query"],
            assistant_message=result.answer,
            citations=[c.chunk_id for c in result.citations],
        )
        get_ltm().log_query(
            session_id=session_id,
            query=state["query"],
            intent=str(intent_enum.value),
            unit=state.get("unit"),
        )

    logger.info(
        "[finalize_node] Query complete in %.1f ms | Status=%s | Self-corrected=%s",
        total_latency_ms, result.status, self_corrected,
    )

    return {**state, "result": result}
