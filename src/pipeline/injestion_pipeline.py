import os
import sys
# Add project root to sys.path
sys.path.append(os.getcwd())

import logging
import uuid
from src.ingestion.pdf_loader import FinancePDFLoader
from src.ingestion.data_cleaner import FinanceDataCleaner
from src.chunking.recursive_chunker import FinanceRecursiveChunker
from src.vectorstore.chromadb_store import ChromaDBStore
from src.embedding.hf_embedder import FinanceEmbedder

# Professional Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self):
        # 1. Initialize our local Brain (Embedder)
        self.embedder = FinanceEmbedder()
        
        # 2. Setup Storage with that Brain to avoid external downloads
        self.vector_store = ChromaDBStore(embedding_function=self.embedder)
        
        # 3. Setup Processing tools
        self.loader = FinancePDFLoader()
        self.cleaner = FinanceDataCleaner()
        self.chunker = FinanceRecursiveChunker()
    
    def run_pipeline(self):
        """
        Coordinates the flow from raw PDF to Vector Database.
        """
        try:
            logger.info("--- 🚀 Starting Ingestion Pipeline (Local-First) ---")

            # Stage 1: Extraction
            raw_docs = self.loader.load_and_save()
            if not raw_docs:
                logger.error("No PDFs found in data/raw/")
                return None
            
            # Stage 2: Processing
            cleaned_docs = self.cleaner.process()
            if not cleaned_docs:
                logger.error("Data cleaning failed.")
                return None

            # Stage 3: Transformation
            chunks_data = self.chunker.create_chunks()
            if not chunks_data:
                logger.error("Chunk generation failed.")
                return None

            # Stage 4: Storage
            logger.info("Indexing chunks into ChromaDB...")
            texts = [c["content"] for c in chunks_data]
            metadatas = [c["metadata"] for c in chunks_data]
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

            # ChromaDB uses 'self.embedder' internally to encode these texts.
            self.vector_store.add_documents(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"--- ✅ Pipeline Completed. Stored {len(texts)} chunks. ---")
            return ids

        except Exception as e:
            logger.error(f"--- ❌ Pipeline Failed: {str(e)} ---")
            return None

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run_pipeline()
