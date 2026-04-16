import logging
import os
import sys
from typing import List, Optional

sys.path.append(os.getcwd())
from dotenv import load_dotenv

load_dotenv()

# Professional Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class RerankingError(Exception):
    """Custom exception raised during reranking operations."""
    pass

class CohereReranker:
    """
    Re-Ranking Engine - Final accuracy filter following the Retrieval Phase.
    Employs the Cohere Rerank v3 API, degrading gracefully to a local Cross-Encoder
    model if cloud APIs are inaccessible.
    """

    LOCAL_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, top_n: int = 3) -> None:
        """
        Initializes the reranking components.

        Args:
            top_n (int, optional): Default number of documents to return. Defaults to 3.
        """
        self.top_n = top_n
        self.client = None
        self.local_model = None
        self.mode: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        """Configures the Cohere client or local fallback based on environment viability."""
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            try:
                import cohere
                self.client = cohere.ClientV2(api_key=api_key)
                self.mode = "cohere"
                logger.info("Reranker Initialized: Cohere API Mode")
                return
            except ImportError:
                logger.error("Cohere SDK not found. Install via `pip install cohere`.")
            except Exception as e:
                logger.warning(f"Cohere API connection failed: {e}. Degrading to local model.")

        # Local Fallback
        try:
            from sentence_transformers import CrossEncoder
            self.local_model = CrossEncoder(self.LOCAL_MODEL_NAME)
            self.mode = "local"
            logger.info(f"Reranker Initialized: Local Cross-Encoder ({self.LOCAL_MODEL_NAME})")
        except ImportError:
            logger.error("sentence_transformers not found. Local reranking disabled.")
        except Exception as e:
            logger.error(f"Failed to intialize local reranker: {e}")

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[str]:
        """
        Scores and re-orders a list of documents based on relevance to the query.

        Args:
            query (str): The informational query.
            documents (List[str]): List of raw document strings gathered from retrievers.
            top_n (Optional[int]): Override default doc count return.

        Returns:
            List[str]: Refined, reranked list of document strings.
            
        Raises:
            RerankingError: Propagated if reranking fully fails without recovering.
        """
        working_top_n = min(top_n or self.top_n, len(documents))

        if not documents:
            logger.debug("Reranking bypassed: Empty document list provided.")
            return []

        if not self.mode:
            logger.error("Active Reranker module missing. Returning raw unstructured results.")
            return documents[:working_top_n]

        logger.info(f"Processing Rerank Task: {len(documents)} candidates down to top {working_top_n} via {self.mode}.")

        if self.mode == "cohere":
            return self._cohere_rerank(query, documents, working_top_n)
        return self._local_rerank(query, documents, working_top_n)

    def _cohere_rerank(self, query: str, documents: List[str], top_n: int) -> List[str]:
        """Orchestrates API calls to Cohere v3 endpoints."""
        try:
            response = self.client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=top_n,
            )
            return [documents[r.index] for r in response.results]
        except Exception as e:
            logger.error(f"Cohere Cloud Rerank Execution Failed: {e}", exc_info=True)
            # Returning raw list as immediate fallback so the pipeline doesn't crash completely.
            return documents[:top_n]

    def _local_rerank(self, query: str, documents: List[str], top_n: int) -> List[str]:
        """Evaluates localized similarity pairs using PyTorch/SentenceTransformers."""
        try:
            # Pair formulation for cross-encoder inference
            pairs = [(query, doc) for doc in documents]
            scores = self.local_model.predict(pairs)
            
            # Map scores to documents and sort
            ranked_pairs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [doc for score, doc in ranked_pairs[:top_n]]
        except Exception as e:
            logger.error(f"Local Cross-Encoder Execution Failed: {e}", exc_info=True)
            return documents[:top_n]

if __name__ == "__main__":
    TEST_QUERY = "What is Wipro's Q4 revenue and operating margin?"
    
    MOCK_DOCS = [
        "Wipro's operating margin in Q4 FY24 stood at 16.1%, up 30 bps sequentially.",
        "Wipro IT Services revenue was $2.63 billion in Q4 FY24, down 4.4% YoY.",
        "Wipro's large deal wins totalled $1.4 billion in Q4 FY24."
    ]

    reranker = CohereReranker(top_n=2)
    results = reranker.rerank(query=TEST_QUERY, documents=MOCK_DOCS)

    print("\n" + "="*70)
    print(f"QUERY: {TEST_QUERY}")
    print("="*70)
    for idx, doc in enumerate(results, 1):
        print(f"[RANK {idx}]: {doc}")
    print("="*70 + "\n")
