import logging
import os
import sys
from typing import List

# Standard Industry Path Setup
sys.path.append(os.getcwd())

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

# Professional Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Advanced Hybrid Retriever combining Dense (Semantic) and Sparse (Keyword) search.
    Implements Reciprocal Rank Fusion (RRF) to merge results.
    """
    def __init__(self):
        # Initializing the specialists
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        logger.info("--- 🚀 Hybrid Retriever is Online with RRF Logic! ---")

    def retrieve(self, query: str, top_k: int = 5):
        """
        Calculates a fused score for chunks from both retrievers.
        """
        logger.info(f"Hybrid retrieval mode: '{query}'")
        
        # Step 1: Gather results (asking for more to allow for better fusion)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2) or []
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k * 2) or []
        
        # Step 2: Reciprocal Rank Fusion (RRF)
        # We reward chunks that appear high in both lists
        combined_scores = {}
        K = 60 # Default constant that balances ranks

        # Voting from Dense Retriever
        for rank, doc in enumerate(dense_results):
            score = 1.0 / (K + rank)
            combined_scores[doc] = combined_scores.get(doc, 0) + score
            
        # Voting from Sparse Retriever
        for rank, doc in enumerate(sparse_results):
            score = 1.0 / (K + rank)
            combined_scores[doc] = combined_scores.get(doc, 0) + score
            
        # Step 3: Final Selection - Sorting by combined weighted score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extracting the content only
        final_docs = [doc for doc, score in sorted_results[:top_k]]
        
        logger.info(f"Successfully fused results using RRF. Returning top {len(final_docs)} candidates.")
        return final_docs

if __name__ == "__main__":
    # Integration Reality Test
    retriever = HybridRetriever()
    
    # Financial Query to test both meaning and keyword
    test_query = "Wipro Q4 financial stats and revenues growth impact"
    
    results = retriever.retrieve(test_query, top_k=5)
    
    print("\n" + "="*75)
    print(f"HYBRID FUSION SEARCH: {test_query}")
    print("="*75)
    
    if results:
        for i, res in enumerate(results):
            print(f"\n[RANK {i+1}]:")
            print(res[:350] + "...")
    else:
        print("No results found. Please check data/processed/chunks.json.")
        
    print("\n" + "="*75)
