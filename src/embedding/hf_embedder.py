import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import EmbeddingFunction, Documents, Embeddings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class FinanceEmbedder(EmbeddingFunction):
    """
    Advanced Embedding class for Finance RAG.
    Implements ChromaDB's EmbeddingFunction interface to prevent default model downloads.
    """
    
    def name(self) -> str:
        return "finance_bge_embedder"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.query_instruction = "Represent this query for retrieving relevant financial information:"
        
        logger.info(f"Loading Local Model: {self.model_name}...")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Local Embedding Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def __call__(self, input: Documents) -> Embeddings:
        """Requirement for ChromaDB EmbeddingFunction interface"""
        # Converting list of strings to list of vectors
        return self.embeddings.embed_documents(input)

    def embed_documents(self, texts: List[str]):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str):
        full_query = f"{self.query_instruction} {query}"
        return self.embeddings.embed_query(full_query)
