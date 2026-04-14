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
    Strictly follows ChromaDB's EmbeddingFunction requirements.
    """
    
    def name(self) -> str:
        return "finance_bge_embedder"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        
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
        """
        Takes a list of strings (Documents/Queries) and returns a list of vectors.
        This is the core method called by ChromaDB.
        """
        # Ensure we always return a list of lists of floats
        vectors = self.embeddings.embed_documents(input)
        return [list(map(float, v)) for v in vectors]

    def embed_documents(self, texts: List[str]):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, input: str):
        # LangChain's embed_query returns a single list[float]
        vector = self.embeddings.embed_query(input)
        return list(map(float, vector))
