"""
SyllAIq — LangGraph Execution Graph
Orchestrates intent routing, hybrid retrieval, relevance grading, reranking, generation, and groundedness verification.
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph

from agents.nodes.finalize_node import finalize_node
from agents.nodes.generate_node import generate_node
from agents.nodes.grade_node import grade_node, route_after_grade
from agents.nodes.intent_node import intent_node
from agents.nodes.rerank_node import rerank_node
from agents.nodes.retrieve_node import retrieve_node
from agents.nodes.rewrite_node import rewrite_node
from agents.nodes.sql_node import sql_node
from agents.nodes.verify_node import verify_node, route_after_verify
from agents.nodes.web_search_node import web_search_node
from agents.state import GraphState, initial_state
from models.documents import Intent
from models.responses import RAGResult
from utils.logger import get_logger

logger = get_logger(__name__)


def route_after_intent(state: GraphState) -> str:
    """
    Routes query based on classified intent:
    - PYQ_ANALYTICS -> direct SQL query
    - Everything else -> full RAG retrieval pipeline
    """
    intent = state.get("intent")
    if intent == Intent.PYQ_ANALYTICS or intent == "analytics":
        logger.info("[route_after_intent] Routing to SQL engine")
        return "sql"
    logger.info("[route_after_intent] Routing to RAG retrieval pipeline")
    return "rewrite"


def build_syllaiq_graph():
    """
    Constructs and compiles the SyllAIq LangGraph agent.
    """
    workflow = StateGraph(GraphState)

    # 1. Add all nodes
    workflow.add_node("intent", intent_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("finalize", finalize_node)

    # 2. Add edges and conditional branches
    workflow.add_edge(START, "intent")

    # Intent routing
    workflow.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "sql": "sql",
            "rewrite": "rewrite",
        },
    )

    # SQL path goes straight to finalize
    workflow.add_edge("sql", "finalize")

    # Retrieval flow
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")

    # Self-RAG Loop 1: Relevance grading & retries / web search
    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "pass": "rerank",
            "retry": "rewrite",
            "web_search": "web_search",
        },
    )

    # Web search fallback flows into reranking
    workflow.add_edge("web_search", "rerank")

    # Generation & Groundedness Verification
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "verify")

    # Self-RAG Loop 2: Groundedness verification & generation retries
    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "grounded": "finalize",
            "retry": "generate",
            "accept": "finalize",
        },
    )

    workflow.add_edge("finalize", END)

    compiled_graph = workflow.compile()
    logger.info("SyllAIq LangGraph agent compiled successfully")
    return compiled_graph


class SyllAIqAgent:
    """
    High-level agent interface wrapping the compiled LangGraph workflow.
    """

    def __init__(self):
        self._graph = build_syllaiq_graph()

    @property
    def graph(self):
        return self._graph

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        unit: Optional[int] = None,
    ) -> RAGResult:
        """
        Executes the agent graph end-to-end for a given query.
        Returns the final RAGResult.
        """
        init_state = initial_state(query=query, session_id=session_id, unit=unit)
        final_state = self._graph.invoke(init_state)

        result = final_state.get("result")
        if result is None:
            # Fallback if somehow result was not populated
            finalize_state = finalize_node(final_state)
            result = finalize_state.get("result")

        return result
