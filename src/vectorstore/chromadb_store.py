import logging
import chromadb
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChromaDBStore:
    """
    Professional Vector Store manager using ChromaDB.
    Updated to handle both automatic and manual embedding query modes.
    """
    
    def __init__(
        self, 
        persist_directory: str = "data/vectorstore/chroma_db",
        collection_name: str = "finance_rag_collection",
        embedding_function: Any = None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"ChromaDB initialized at {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents: List[str], ids: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        try:
            self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
        except Exception as e:
            logger.error(f"Add error: {str(e)}")
            raise

    def query(self, query_text: str = None, query_embeddings: List[List[float]] = None, n_results: int = 4) -> Dict[str, Any]:
        """
        Supports querying via text (automatic) or pre-calculated embeddings (manual).
        Manual is more robust for custom local models.
        """
        try:
            if query_embeddings:
                results = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
            return results
        except Exception as e:
            logger.error(f"Error during ChromaDB query: {str(e)}")
            return {}

    def get_collection_count(self) -> int:
        return self.collection.count()
