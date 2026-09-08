"""
Groundedness verification node.
Checks if generated answer is grounded in retrieved context using NLI verification.
"""

from __future__ import annotations

from typing import Optional

from agents.state import GraphState
from retrieval.nli_grader import NLIGrader
from utils.logger import get_logger

logger = get_logger(__name__)

_grader: Optional[NLIGrader] = None

# Max times we'll retry generation before accepting the answer anyway
MAX_GENERATION_RETRIES = 1


def get_grader() -> NLIGrader:
    global _grader
    if _grader is None:
        _grader = NLIGrader()
    return _grader


def verify_node(state: GraphState) -> GraphState:
    """
    Verifies that the generated answer is grounded in the retrieved docs.
    Uses NLIGrader.check_groundedness() — which is fail-secure (False, 0.0 on missing model).
    Sets state['is_grounded'].
    """
    answer = state.get("answer", "")
    ranked_docs = state.get("ranked_docs", [])
    top_docs = [doc for doc, _ in ranked_docs[:3]]

    if not answer.strip():
        logger.warning("[verify_node] Empty answer — marking as ungrounded")
        return {**state, "is_grounded": False}

    is_grounded, score = get_grader().check_groundedness(
        answer=answer,
        documents=top_docs,
    )

    logger.info(
        "[verify_node] is_grounded=%s nli_score=%.3f gen_attempts=%d",
        is_grounded, score, state.get("generation_attempts", 0),
    )

    return {**state, "is_grounded": is_grounded}


def route_after_verify(state: GraphState) -> str:
    """
    Conditional edge after verify node.
    Returns one of: 'grounded' | 'retry' | 'accept'
    """
    is_grounded = state.get("is_grounded", False)
    gen_attempts = state.get("generation_attempts", 0)

    if is_grounded:
        logger.info("[route_after_verify] GROUNDED — proceeding to finalize")
        return "grounded"

    if gen_attempts <= MAX_GENERATION_RETRIES:
        logger.info("[route_after_verify] RETRY — ungrounded, attempt=%d", gen_attempts)
        return "retry"

    logger.warning(
        "[route_after_verify] ACCEPT — max retries (%d) hit, accepting with warning",
        gen_attempts,
    )
    return "accept"
