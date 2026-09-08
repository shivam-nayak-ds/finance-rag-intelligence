"""SQL analytics node — routes analytical queries directly to SQLQueryEngine."""

from __future__ import annotations

from typing import Optional

from agents.state import GraphState
from tools.sql_query_engine import SQLQueryEngine
from utils.logger import get_logger

logger = get_logger(__name__)

_sql_engine: Optional[SQLQueryEngine] = None


def get_sql_engine() -> SQLQueryEngine:
    global _sql_engine
    if _sql_engine is None:
        _sql_engine = SQLQueryEngine()
    return _sql_engine


def sql_node(state: GraphState) -> GraphState:
    """
    Handles PYQ_ANALYTICS intent by querying the SQLite PYQ database.
    Bypasses the full RAG pipeline for structured analytics queries.
    Sets state['result'] directly so finalize_node passes it through.
    """
    query = state["query"]
    logger.info("[sql_node] Routing to SQLQueryEngine: %r", query[:80])

    sql_result = get_sql_engine().execute_and_format(query)

    logger.info(
        "[sql_node] SQL status=%s intent=%s",
        sql_result.status, sql_result.intent,
    )

    return {
        **state,
        "answer": sql_result.answer,
        "citations": sql_result.citations,
        "confidence": sql_result.confidence,
        "confidence_score": 1.0 if sql_result.status == "success" else 0.0,
        "is_grounded": True,
        "result": sql_result,
    }
