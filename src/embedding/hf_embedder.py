import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Logger setup: Ye humein terminal mein batayega ki code kya kar raha hai
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinanceEmbedder:
    """
    Advanced Embedding class for Finance RAG.
    Motive: Text ko numerical vectors mein badalna.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        
        # BGE models better results dete hain jab hum unhe batate hain ki wo kya dhund rahe hain
        self.query_instruction = "Represent this query for retrieving relevant financial information:"
        
        logger.info(f"Loading Model: {self.model_name}...")
        
        try:
            # HuggingFace se model load karna
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'}, # 'cpu' par chalega
                encode_kwargs={'normalize_embeddings': True} # Vectors ko standard size ka banayega
            )
            logger.info("Model load ho gaya!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]):
        """Saare data chunks ko vectors mein badalna"""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str):
        """User ke question ko vector mein badalna (instruction ke saath)"""
        full_query = f"{self.query_instruction} {query}"
        return self.embeddings.embed_query(full_query)

if __name__ == "__main__":
    # Test karne ke liye
    embedder = FinanceEmbedder()
    print("Finance Embedder is ready to use!")
