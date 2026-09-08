"""SyllAIq end-to-end RAG pipeline and agent orchestrator."""

from __future__ import annotations

import time
from typing import Optional

from agents.graph import SyllAIqAgent
from config.settings import RERANK_TOP_N
from generation.llm_chain import LLMChain
from generation.memory import ConversationMemory, LongTermMemory, get_memory, get_long_term_memory
from models.documents import ConfidenceLevel, Intent
from models.responses import RAGResult
from retrieval.cohere_reranker import CohereReranker
from retrieval.hybrid_retriever import HybridRetriever
from tools.confidence_scorer import ConfidenceScorer
from tools.query_rewriter import QueryRewriter
from tools.sql_query_engine import SQLQueryEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for SyllAIq.
    Supports both Agentic Self-RAG mode (powered by LangGraph)
    and linear legacy mode with lazy component initialization.
    """

    def __init__(
        self,
        rewriter: Optional[QueryRewriter] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CohereReranker] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        llm_chain: Optional[LLMChain] = None,
        memory: Optional[ConversationMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        sql_engine: Optional[SQLQueryEngine] = None,
        agent: Optional[SyllAIqAgent] = None,
        top_n: int = RERANK_TOP_N,
    ) -> None:
        self._rewriter = rewriter
        self._retriever = retriever
        self._reranker = reranker
        self._confidence_scorer = confidence_scorer
        self._llm_chain = llm_chain
        self._memory = memory
        self._long_term = long_term
        self._sql_engine = sql_engine
        self._agent = agent
        self.top_n = top_n

    @property
    def agent(self) -> SyllAIqAgent:
        if self._agent is None:
            self._agent = SyllAIqAgent()
        return self._agent

    @property
    def rewriter(self) -> QueryRewriter:
        if self._rewriter is None:
            self._rewriter = QueryRewriter()
        return self._rewriter

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    @property
    def reranker(self) -> CohereReranker:
        if self._reranker is None:
            self._reranker = CohereReranker()
        return self._reranker

    @property
    def confidence_scorer(self) -> ConfidenceScorer:
        if self._confidence_scorer is None:
            self._confidence_scorer = ConfidenceScorer()
        return self._confidence_scorer

    @property
    def llm_chain(self) -> LLMChain:
        if self._llm_chain is None:
            self._llm_chain = LLMChain()
        return self._llm_chain

    @property
    def memory(self) -> ConversationMemory:
        if self._memory is None:
            self._memory = get_memory()
        return self._memory

    @property
    def long_term(self) -> LongTermMemory:
        if self._long_term is None:
            self._long_term = get_long_term_memory()
        return self._long_term

    @property
    def sql_engine(self) -> SQLQueryEngine:
        if self._sql_engine is None:
            self._sql_engine = SQLQueryEngine()
        return self._sql_engine

    # ──────────────────────────────────────────
    # Intent classification (keyword heuristic)
    # ──────────────────────────────────────────

    def _classify_intent(self, query: str) -> str:
        q = query.lower()

        # Check for analytical / count / table queries first
        analytical_keywords = [
            "how many", "count", "kitne", "kitni", "total questions",
            "list all", "table of", "marks analysis", "frequency",
            "most asked", "least asked", "repeat count", "year wise",
            "distribution", "statistics", "analytics",
        ]
        if any(k in q for k in analytical_keywords):
            return Intent.PYQ_ANALYTICS

        pyq_keywords = ["pyq", "previous year", "purane question", "exam mein", "kitni baar", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
        importance_keywords = ["important", "important topics", "kya padhein", "kitna important", "frequently", "baar baar"]
        syllabus_keywords = ["syllabus", "unit", "kya kya aata hai", "course", "rgpv syllabus"]

        if any(k in q for k in pyq_keywords):
            return Intent.PYQ_RETRIEVAL
        if any(k in q for k in importance_keywords):
            return Intent.TOPIC_IMPORTANCE
        if any(k in q for k in syllabus_keywords):
            return Intent.SYLLABUS_LOOKUP
        return Intent.CONCEPT_EXPLANATION

    # ──────────────────────────────────────────
    # Main ask method
    # ──────────────────────────────────────────

    def ask(
        self,
        query: str,
        session_id: Optional[str] = None,
        unit: Optional[int] = None,
        use_agent: bool = True,
    ) -> RAGResult:
        """
        Process a student query end-to-end.

        Args:
            query: Student's question (English/Hinglish).
            session_id: Optional session ID for conversation memory.
            unit: Optional syllabus unit filter (1-5).
            use_agent: If True (default), run the LangGraph Agentic Self-RAG loop.
                       If False, run the linear fallback pipeline.

        Returns:
            RAGResult with answer, citations, confidence, and latency.
        """
        if use_agent:
            try:
                logger.info("=== RAGPipeline.ask() [Agentic Self-RAG] ===")
                logger.info("Query: %r | session_id=%s | unit=%s", query[:80], session_id, unit)
                return self.agent.run(query=query, session_id=session_id, unit=unit)
            except Exception as exc:
                logger.error("Agentic graph execution error: %s — falling back to linear pipeline", exc)

        return self.ask_linear(query=query, session_id=session_id, unit=unit)

    def ask_linear(
        self,
        query: str,
        session_id: Optional[str] = None,
        unit: Optional[int] = None,
    ) -> RAGResult:
        """
        Legacy linear pipeline execution.
        """
        pipeline_start = time.perf_counter()
        logger.info("=== RAGPipeline.ask_linear() ===")
        logger.info("Query: %r | session_id=%s | unit=%s", query[:80], session_id, unit)

        # Step 1: Save user turn to short-term memory
        if session_id:
            self.memory.add_user_turn(session_id, query)

        # Step 2: Intent classification
        intent = self._classify_intent(query)
        logger.info("Intent: %s", intent)

        # Step 2b: Direct routing for analytical / statistical queries to SQL
        if intent == Intent.PYQ_ANALYTICS:
            logger.info("Routing to SQLQueryEngine for analytical query")
            result = self.sql_engine.execute_and_format(query)
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            result.total_latency_ms = total_ms

            if session_id:
                self.memory.add_assistant_turn(session_id, result.answer, intent=intent)
                self.long_term.log_query(
                    session_id=session_id,
                    query=query,
                    intent=intent,
                    confidence=str(result.confidence),
                )
            return result

        # Step 3: Query rewriting
        context_hint = self.memory.get_context_summary(session_id) if session_id else ""
        rewritten_query = self.rewriter.rewrite(query, context=context_hint)
        logger.info("Rewritten query: %r", rewritten_query[:80])

        # Step 4: Hybrid retrieval
        t_ret = time.perf_counter()
        candidates = self.retriever.retrieve(
            query=rewritten_query,
            intent=intent,
            unit=unit,
        )
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        if not candidates:
            logger.warning("No candidates retrieved for query: %r", query[:60])
            return RAGResult(
                answer="Maafi karo, is topic par koi relevant content knowledge base mein nahi mila.",
                citations=[],
                confidence=ConfidenceLevel.LOW,
                intent=intent,
                retrieval_failed=True,
                retrieval_time_ms=retrieval_ms,
                total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
                status="partial",
            )

        # Step 5: Reranking
        t_rerank = time.perf_counter()
        ranked_docs = self.reranker.rerank(query=rewritten_query, documents=candidates, top_n=self.top_n)
        reranking_ms = (time.perf_counter() - t_rerank) * 1000

        # Step 6: Confidence scoring
        confidence_level, confidence_score = self.confidence_scorer.score(ranked_docs)
        warning = self.confidence_scorer.warning_message(confidence_level)

        # Step 7: Extract top documents for generation
        top_docs = [doc for doc, _ in ranked_docs]

        # Step 8: Generate answer
        all_msgs = self.memory.get_history_messages(session_id) if session_id else []
        history_msgs = all_msgs[:-1] if len(all_msgs) > 1 else []
        personalization = (
            self.long_term.get_personalization_hint(session_id)
            if session_id
            else None
        )

        result = self.llm_chain.generate(
            query=query,
            documents=top_docs,
            ranked_docs=ranked_docs,
            intent=intent,
            confidence=confidence_level,
            confidence_score=confidence_score,
            warning=warning,
            retrieval_time_ms=retrieval_ms,
            reranking_time_ms=reranking_ms,
            history_messages=history_msgs,
            personalization_hint=personalization,
        )

        # Step 9: Update both memories
        if session_id:
            self.memory.add_assistant_turn(session_id, result.answer, intent=intent)
            top_topic = top_docs[0].topic if top_docs else None
            top_unit = top_docs[0].unit if top_docs else None
            self.long_term.log_query(
                session_id=session_id,
                query=query,
                intent=intent,
                topic=top_topic,
                unit=top_unit,
                confidence=str(confidence_level),
            )
            if confidence_level == ConfidenceLevel.LOW and top_topic:
                self.long_term.mark_weak_area(
                    session_id=session_id,
                    topic=top_topic,
                    unit=top_unit,
                )

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        result.total_latency_ms = total_ms
        logger.info(
            "Pipeline complete: intent=%s confidence=%s latency=%.0fms tokens=%d",
            intent, confidence_level, total_ms, result.total_tokens,
        )
        return result
