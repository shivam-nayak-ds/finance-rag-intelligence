import logging
import os
import sys

# Standard Industry practice for local paths
sys.path.append(os.getcwd())

from src.vectorstore.chromadb_store import ChromaDBStore
from src.embedding.hf_embedder import FinanceEmbedder

# Professional logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DenseRetriever:
    """
    Expert Detective: Searches for documents using semantic meaning (Context).
    Manual Vector Injection mode for 100% reliability.
    """
    def __init__(self):
        # 1. Loading Local Brain
        self.embedder = FinanceEmbedder()
        
        # 2. Connecting to our Knowledge Base
        self.vector_store = ChromaDBStore(embedding_function=self.embedder)
        logger.info("--- 🔍 Dense Retriever (Semantic) is Initialized ---")

    def retrieve(self, query: str, top_k: int = 5):
        """
        Takes a text query, converts it to vector manually, and finds the best matching chunks.
        """
        try:
            logger.info(f"Retrieving top {top_k} results for: '{query}'")
            
            # Step 1: Manually calculate query embedding
            query_vector = self.embedder.embed_query(query)
            
            # Step 2: Query ChromaDB using the vector directly
            results = self.vector_store.query(
                query_embeddings=[query_vector], # Wrap in list
                n_results=top_k
            )
            
            if not results or not results['documents'] or len(results['documents'][0]) == 0:
                logger.warning("No relevant context found in the database.")
                return []
            
            retrieved_chunks = results['documents'][0]
            logger.info(f"Successfully retrieved {len(retrieved_chunks)} semantic chunks.")
            
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []

if __name__ == "__main__":
    # REAL WORLD TEST
    retriever = DenseRetriever()
    
    # Let's ask a financial question based on your uploaded PDFs
    test_query = "What is the total revenue or income mentioned in the documents?"
    
    context = retriever.retrieve(test_query, top_k=3)
    
    print("\n" + "="*60)
    print(f"QUESTION: {test_query}")
    print("="*60)
    
    if context:
        for i, chunk in enumerate(context):
            print(f"\n[RANK {i+1} CHUNK]:")
            print(chunk[:400] + "...") 
    else:
        print("No results found. Please ensure the pipeline has been run first.")
    
    print("\n" + "="*60)
