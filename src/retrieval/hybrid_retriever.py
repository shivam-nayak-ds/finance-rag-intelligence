import logging
import os
import sys
from typing import List, Dict, Any, Optional

# Standard Industry Path Setup
sys.path.append(os.getcwd())

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

# Professional Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridRetrieverError(Exception):
    """Custom exception raised during failed hybrid retrieval fusion."""
    pass

class HybridRetriever:
    """
    Advanced Hybrid Retriever Architecture combining Dense (Semantic) and Sparse (Keyword) search mechanisms.
    Leverages Reciprocal Rank Fusion (RRF) to merge and rank results optimally.
    """

    def __init__(self, rrf_k: int = 60) -> None:
        """
        Initializes the dense and sparse specialized retrievers.
        
        Args:
            rrf_k (int): Constant used in the RRF formula to penalize extreme outlier ranks. Default is 60.
        """
        self.rrf_k = rrf_k
        logger.info("Initializing Hybrid Retrieval Modules...")
        try:
            self.dense_retriever = DenseRetriever()
            self.sparse_retriever = SparseRetriever()
            logger.info("✅ Hybrid Retriever sequence active with RRF capabilities.")
        except Exception as e:
            logger.critical(f"Failed to mount sub-retrievers: {e}")
            raise HybridRetrieverError("Sub-retriever mounting sequence failed.") from e

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Performs dual-retrieval and merges the output via RRF algorithm.

        Args:
            query (str): User's primary search string.
            top_k (int): Exact number of top retrieved documents required.

        Returns:
            List[str]: Combined and mathematically sorted list of the best chunk textual content.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to Hybrid Retriever.")
            return []

        logger.info(f"Hybrid retrieval initiated for query: '{query}'")
        
        try:
            # Step 1: Parallel or Sequential fetch (Gathering more than `top_k` for better fusion)
            fetch_k = top_k * 2
            
            # Sub-retrievers should be resilient and return empty lists rather than crashing
            dense_results: List[str] = self.dense_retriever.retrieve(query, top_k=fetch_k) or []
            sparse_results: List[str] = self.sparse_retriever.retrieve(query, top_k=fetch_k) or []
            
            # Step 2: Applying Reciprocal Rank Fusion (RRF) algorithm
            combined_scores: Dict[str, float] = {}

            # Voting mechanism from Dense Retriever
            for rank, doc in enumerate(dense_results):
                # Using 1 / (k + rank) where rank is 0-indexed
                score = 1.0 / (self.rrf_k + rank)
                combined_scores[doc] = combined_scores.get(doc, 0.0) + score
                
            # Voting mechanism from Sparse Retriever
            for rank, doc in enumerate(sparse_results):
                score = 1.0 / (self.rrf_k + rank)
                combined_scores[doc] = combined_scores.get(doc, 0.0) + score
                
            # Step 3: Global Sorting by combined fusion score
            sorted_fused_results = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
            
            # Stripping out the scores to return purely document strings
            final_docs = [doc for doc, score in sorted_fused_results[:top_k]]
            
            logger.info(f"RRF Fusion Complete. Distilled down to top {len(final_docs)} candidate chunks.")
            return final_docs

        except Exception as e:
            logger.error(f"Hybrid Retrieval Engine failed to process query: {e}", exc_info=True)
            return []

if __name__ == "__main__":
    # Regression test block
    retriever = HybridRetriever()
    test_query = "Wipro Q4 financial stats and revenues"
    
    results = retriever.retrieve(test_query, top_k=5)
    
    print("\n" + "="*75)
    print(f"HYBRID FUSION SEARCH TEST: {test_query}")
    print("="*75)
    
    if results:
        for idx, res in enumerate(results, 1):
            print(f"\n[RANK {idx}]:")
            print(res[:350] + ("..." if len(res) > 350 else ""))
    else:
        print("No valid results found. Verify indexing state.")
        
    print("\n" + "="*75)
