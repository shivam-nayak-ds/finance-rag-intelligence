"""
Pipeline & Agent Unit Tests for SyllAIq.
Tests routing, LangGraph graph integration, and fallback mechanisms.
"""

import pytest

import unittest

from agents.graph import SyllAIqAgent, build_syllaiq_graph
from models.documents import Intent
from pipeline.rag_pipeline import RAGPipeline


class TestPipelineUnit(unittest.TestCase):
    """Test RAGPipeline configuration and routing."""

    def test_pipeline_instantiation(self):
        pipeline = RAGPipeline()
        assert pipeline is not None
        assert pipeline.top_n == 5

    def test_intent_classification(self):
        pipeline = RAGPipeline()
        # Analytical
        assert pipeline._classify_intent("How many questions in 2023?") == Intent.PYQ_ANALYTICS
        assert pipeline._classify_intent("Table of repeat topics") == Intent.PYQ_ANALYTICS
        # Conceptual
        assert pipeline._classify_intent("Explain Bankers algorithm") == Intent.CONCEPT_EXPLANATION
        # Syllabus
        assert pipeline._classify_intent("What is in Unit 2 syllabus?") == Intent.SYLLABUS_LOOKUP

    def test_agent_graph_compilation(self):
        graph = build_syllaiq_graph()
        assert graph is not None

    def test_agent_wrapper(self):
        agent = SyllAIqAgent()
        assert agent.graph is not None
