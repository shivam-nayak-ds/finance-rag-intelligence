import logging
import chromadb
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChromaDBStore:
    """
    Professional Vector Store manager using ChromaDB.
    Handles persistent storage and semantic retrieval of document chunks.
    """
    
    def __init__(
        self, 
        persist_directory: str = "data/vectorstore/chroma_db",
        collection_name: str = "finance_rag_collection"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize Persistent Client
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
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

if __name__ == "__main__":
    # Test block to verify initialization
    logging.basicConfig(level=logging.INFO)
    try:
        store = ChromaDBStore()
        print(f"Connection Successful. Current document count: {store.get_collection_count()}")
    except Exception as e:
        print(f"Setup Failed: {e}")
