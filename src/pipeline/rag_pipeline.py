import logging
import os
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Ensure custom modules are found when running standalone
sys.path.append(os.getcwd())

# Core module imports
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cohere_reranker import CohereReranker
from src.generation.llm_chain import LLMChain

# Professional Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Standardized response object for the RAG Pipeline."""
    answer: str
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None
    status: str = "success"

class FinanceRAGPipeline:
    """
    Enterprise-grade Execution Pipeline for Advanced RAG.
    Orchestrates Retrieval -> Reranking -> Generation with fault tolerance.
    """

    def __init__(self) -> None:
        logger.info("Initializing Finance RAG Pipeline Architecture...")

        try:
            # Lazy initialization or direct instantiation of core engines
            self.retriever = HybridRetriever()
            self.reranker = CohereReranker()
            self.generator = LLMChain()
            logger.info("✅ All core modules loaded and validated successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize core RAG modules. Trace: {e}")
            raise RuntimeError("Pipeline Initialization Engine Error") from e

    def ask(self, query: str) -> RAGResponse:
        """
        Executes the end-to-end RAG workflow with resilient error handling.
        """
        if not query or not query.strip():
            return RAGResponse(answer="", error="Query cannot be empty.", status="failed")

        logger.info(f"\n🤔 Processing User Query: '{query}'")

        try:
            # ---------------------------------------------------------
            # STEP 1: Retrieval (Hybrid Search)
            # ---------------------------------------------------------
            logger.info("Step 1/3: Commencing Hybrid Retrieval...")
            raw_results = self.retriever.retrieve(query, top_k=20)
            
            if not raw_results:
                logger.warning("Retrieval returned empty. No context found.")
                return RAGResponse(
                    answer="I couldn't find relevant financial data in the provided documents.",
                    status="partial_success"
                )

            # Safely handle the format (since Hybrid Retriever returns list of strings directly)
            retrieved_chunks = raw_results

            # ---------------------------------------------------------
            # STEP 2: Reranking (Refining top results with fallback)
            # ---------------------------------------------------------
            logger.info("Step 2/3: Commencing Cohere Reranking...")
            try:
                # Top_n exposed as a variable for easy configuration later
                top_contexts = 5
                reranked_chunks = self.reranker.rerank(query, retrieved_chunks, top_n=top_contexts)
            except Exception as rerank_err:
                # Graceful Degradation: If AI Cohere fails, fallback to basic retrieval results
                logger.warning(f"Reranking engine failed ({rerank_err}), falling back to base retriever metrics.")
                reranked_chunks = retrieved_chunks[:5]

            # ---------------------------------------------------------
            # STEP 3: Generation (LLM Response)
            # ---------------------------------------------------------
            logger.info("Step 3/3: Generating Grounded LLM Response...")
            final_answer = self.generator.generate(query=query, context_chunks=reranked_chunks)

            # Validate LLM output
            if not final_answer or "Error" in final_answer:
                raise ValueError("LLM failed to generate a coherent response.")

            logger.info("✅ Pipeline execution completed successfully.")
            return RAGResponse(
                answer=final_answer,
                sources=reranked_chunks
            )

        except Exception as e:
            logger.error(f"❌ Critical Pipeline Failure: {str(e)}", exc_info=True)
            return RAGResponse(
                answer="Our systems are currently facing an issue processing your request. Please try again.",
                error=str(e),
                status="failed"
            )

# Standalone execution for testing
if __name__ == "__main__":
    pipeline = FinanceRAGPipeline()
    
    test_query = "What is Wipro's revenue in Q4 FY24?"
    response = pipeline.ask(test_query)
    
    print("\n" + "="*70)
    print(f"QUESTION: {test_query}")
    print("="*70)
    print(f"STATUS: {response.status.upper()}")
    if response.status == "success":
        print(f"ANSWER:\n{response.answer}")
        print("="*70)
        print(f"SOURCES USED: {len(response.sources)} filtered chunks.")
    else:
        print(f"ERROR LOG: {response.error}")
    print("="*70 + "\n")
