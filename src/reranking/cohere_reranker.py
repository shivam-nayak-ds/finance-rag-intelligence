import logging
import os
import sys
from typing import List, Optional

sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CohereReranker:
    """
    Re-Ranking Engine — Final quality gate after Hybrid Retrieval.
    Uses Cohere Rerank API (if key available) or local Cross-Encoder as fallback.
    """

    LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, top_n: int = 3):
        self.top_n = top_n
        self.client = None
        self.local_model = None
        self.mode = None
        self._initialize()

    def _initialize(self):
        """Sets up Cohere API client or falls back to local Cross-Encoder."""
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            try:
                import cohere
                self.client = cohere.Client(api_key=api_key)
                self.mode = "cohere"
                logger.info("--- 🎯 Re-Ranker Online: Cohere API Mode ---")
                return
            except Exception as e:
                logger.warning(f"Cohere failed: {e}. Falling back to local model.")

        try:
            from sentence_transformers import CrossEncoder
            self.local_model = CrossEncoder(self.LOCAL_MODEL)
            self.mode = "local"
            logger.info("--- 🎯 Re-Ranker Online: Local Cross-Encoder Mode ---")
        except Exception as e:
            logger.error(f"Re-Ranker initialization failed: {e}")

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[str]:
        """
        Re-ranks candidate documents and returns the most relevant ones.
        """
        top_n = top_n or self.top_n

        if not documents:
            logger.warning("No documents to re-rank.")
            return []

        if not self.mode:
            logger.error("No backend available. Returning raw candidates.")
            return documents[:top_n]

        logger.info(f"Re-ranking {len(documents)} candidates → top {top_n} | mode={self.mode}")

        if self.mode == "cohere":
            return self._cohere_rerank(query, documents, top_n)
        return self._local_rerank(query, documents, top_n)

    def _cohere_rerank(self, query: str, documents: List[str], top_n: int) -> List[str]:
        try:
            response = self.client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=top_n,
            )
            return [documents[r.index] for r in response.results]
        except Exception as e:
            logger.error(f"Cohere rerank failed: {e}")
            return documents[:top_n]

    def _local_rerank(self, query: str, documents: List[str], top_n: int) -> List[str]:
        try:
            scores = self.local_model.predict([(query, doc) for doc in documents])
            ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[:top_n]]
        except Exception as e:
            logger.error(f"Local rerank failed: {e}")
            return documents[:top_n]


if __name__ == "__main__":
    TEST_QUERY = "What is Wipro's Q4 revenue and operating margin?"

    MOCK_DOCS = [
        "Wipro's operating margin in Q4 FY24 stood at 16.1%, up 30 bps sequentially.",
        "Wipro IT Services revenue was $2.63 billion in Q4 FY24, down 4.4% YoY.",
        "Wipro's large deal wins totalled $1.4 billion in Q4 FY24.",
        "TCS reported 9% headcount growth in Q4 FY24.",
        "Global commodity prices rose impacting manufacturing supply chains.",
    ]

    reranker = CohereReranker(top_n=3)
    results = reranker.rerank(query=TEST_QUERY, documents=MOCK_DOCS)

    print("\n" + "="*65)
    print(f"QUERY: {TEST_QUERY}")
    print("="*65)
    for i, doc in enumerate(results, 1):
        print(f"\n[RANK {i}]: {doc}")
    print("\n" + "="*65)
