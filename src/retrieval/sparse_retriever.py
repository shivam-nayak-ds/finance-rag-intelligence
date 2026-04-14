import logging
import json
import os
import sys
from rank_bm25 import BM25Okapi

# Standard Industry Path Setup
sys.path.append(os.getcwd())

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparseRetriever:
    """
    Keyword-based search engine using BM25 algorithm.
    Used for high-precision retrieval of financial terms, numbers, and dates.
    """
    def __init__(self, chunks_path: str = "data/processed/chunks.json"):
        self.chunks_path = chunks_path
        self.bm25 = None
        self.corpus = []
        self._initialize_bm25()

    def _initialize_bm25(self):
        """
        Loads all processed chunks and tokenizes them for the BM25 indexer.
        """
        try:
            if not os.path.exists(self.chunks_path):
                logger.error(f"Chunks file {self.chunks_path} not found. Please run ingestion first.")
                return

            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extracting only the text content for the corpus
            self.corpus = [item['content'] for item in data]
            
            # Simple tokenization: Lowercase and split into words
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            
            # Initialize BM25Okapi engine
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"--- 📝 Sparse Retriever initialized with {len(self.corpus)} chunks! ---")

        except Exception as e:
            logger.error(f"Failed to initialize Sparse Retriever: {str(e)}")

    def retrieve(self, query: str, top_k: int = 5):
        """
        Takes a query, tokenizes it, and returns the top relevant chunks based on keyword matching.
        """
        if not self.bm25:
            logger.error("BM25 index not available.")
            return []
            
        logger.info(f"Sparse searching for: '{query}'")
        
        # Tokenize query to match corpus format
        tokenized_query = query.lower().split()
        
        # Get top matching documents
        top_docs = self.bm25.get_top_n(tokenized_query, self.corpus, n=top_k)
        
        return top_docs

if __name__ == "__main__":
    # Integration Test
    retriever = SparseRetriever()
    
    # Financial query for keyword testing
    test_query = "Wipro revenues impact FS client quarter"
    
    results = retriever.retrieve(test_query, top_k=3)
    
    print("\n" + "="*60)
    print(f"KEYWORD QUERY: {test_query}")
    print("="*60)
    
    if results:
        for i, res in enumerate(results):
            print(f"\n[BM25 RANK {i+1}]:")
            print(res[:350] + "...")
    else:
        print("No matches found. Check if your chunks contain these keywords.")
    
    print("\n" + "="*60)
