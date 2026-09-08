"""
Unit tests for SyllAIq LangGraph Agent & Self-RAG Nodes.
"""

import pytest

from agents.graph import build_syllaiq_graph, route_after_intent
from agents.nodes.finalize_node import finalize_node
from agents.nodes.grade_node import route_after_grade
from agents.nodes.intent_node import intent_node
from agents.nodes.verify_node import route_after_verify
from agents.state import initial_state
from models.documents import Document, Intent, SourceType


class TestGraphState:
    """Test state initialization and defaults."""

    def test_initial_state_defaults(self):
        state = initial_state(query="What is page fault?", session_id="test_sess_1", unit=3)
        assert state["query"] == "What is page fault?"
        assert state["session_id"] == "test_sess_1"
        assert state["unit"] == 3
        assert state["retrieval_attempts"] == 0
        assert state["generation_attempts"] == 0
        assert state["is_grounded"] is False
        assert state["web_search_used"] is False


class TestRoutingLogic:
    """Test conditional edge functions."""

    def test_route_after_intent_sql(self):
        state = initial_state(query="How many questions in unit 2?")
        state["intent"] = Intent.PYQ_ANALYTICS
        route = route_after_intent(state)
        assert route == "sql"

    def test_route_after_intent_concept(self):
        state = initial_state(query="Explain Belady's anomaly")
        state["intent"] = Intent.CONCEPT_EXPLANATION
        route = route_after_intent(state)
        assert route == "rewrite"

    def test_route_after_grade_pass(self):
        doc1 = Document(chunk_id="c1", text="text1", source_type=SourceType.TEXTBOOK)
        doc2 = Document(chunk_id="c2", text="text2", source_type=SourceType.TEXTBOOK)
        state = initial_state(query="test")
        state["relevant_docs"] = [doc1, doc2]
        state["retrieval_attempts"] = 1
        assert route_after_grade(state) == "pass"

    def test_route_after_grade_retry(self):
        doc1 = Document(chunk_id="c1", text="text1", source_type=SourceType.TEXTBOOK)
        state = initial_state(query="test")
        state["relevant_docs"] = [doc1]  # < MIN_RELEVANT_DOCS (2)
        state["retrieval_attempts"] = 1   # < 2
        assert route_after_grade(state) == "retry"

    def test_route_after_grade_web_search(self):
        state = initial_state(query="test")
        state["relevant_docs"] = []
        state["retrieval_attempts"] = 2   # exhausted 2 attempts
        assert route_after_grade(state) == "web_search"

    def test_route_after_verify_grounded(self):
        state = initial_state(query="test")
        state["is_grounded"] = True
        state["generation_attempts"] = 1
        assert route_after_verify(state) == "grounded"

    def test_route_after_verify_retry(self):
        state = initial_state(query="test")
        state["is_grounded"] = False
        state["generation_attempts"] = 1   # <= MAX_GENERATION_RETRIES (1)
        assert route_after_verify(state) == "retry"

    def test_route_after_verify_accept_with_warning(self):
        state = initial_state(query="test")
        state["is_grounded"] = False
        state["generation_attempts"] = 2   # > 1
        assert route_after_verify(state) == "accept"


class TestIntentNode:
    """Test keyword heuristic intent classification node."""

    def test_pyq_analytics_intent(self):
        state = initial_state(query="How many total questions were asked from unit 1?")
        next_state = intent_node(state)
        assert next_state["intent"] == Intent.PYQ_ANALYTICS

    def test_concept_intent(self):
        state = initial_state(query="Explain banker's algorithm with safety check")
        next_state = intent_node(state)
        assert next_state["intent"] == Intent.CONCEPT_EXPLANATION


class TestFinalizeNode:
    """Test packaging state into RAGResult."""

    def test_finalize_node_success(self):
        state = initial_state(query="What is semaphores?")
        state["answer"] = "A semaphore is a synchronization variable."
        state["is_grounded"] = True
        state["confidence"] = "high"
        state["confidence_score"] = 0.92

        final_state = finalize_node(state)
        result = final_state["result"]
        assert result is not None
        assert result.answer == "A semaphore is a synchronization variable."
        assert result.confidence == "high"
        assert result.grounding_score == 1.0


class TestGraphCompilation:
    """Test that the full LangGraph state graph compiles cleanly."""

    def test_graph_compile(self):
        graph = build_syllaiq_graph()
        assert graph is not None
