import logging
import chromadb
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChromaDBStore:
    """
    Professional Vector Store manager using ChromaDB.
    Customized to take an external embedding function for better accuracy.
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
        
        # Initialize Persistent Client
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Using custom embedding function if provided
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"ChromaDB initialized at {persist_directory} with collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents: List[str], ids: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Adds text documents to the vector store.
        """
        try:
            logger.info(f"Adding {len(documents)} chunks to ChromaDB.")
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            logger.info("Successfully added documents to ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            raise

    def query(self, query_text: str, n_results: int = 4) -> Dict[str, Any]:
        """
        Performs semantic search to retrieve the most relevant document chunks.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error during ChromaDB query: {str(e)}")
            return {}

    def get_collection_count(self) -> int:
        """Returns the total number of items in the collection."""
        return self.collection.count()
