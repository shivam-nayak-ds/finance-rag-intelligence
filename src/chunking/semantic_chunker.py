import json 
import logging
import pathlib as path
from typing import List , Dict
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_huggingface import HuggingFaceEmbeddings 

load_dotenv()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinanceSemanticChunker:
    def __init__(self, input_file: str = "data/processed/cleaned_data.json", output_file: str = "data/processed/semantic_chunks.json"):
        self.input_file = path.Path(input_file)
        self.output_file = path.Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Free model from HuggingFace
        logger.info("Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
        self.embedding = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
            
        # Semantic Chunker setup
        logger.info("Initializing Semantic Chunker...")
        # SHI KIYA: Argument name sahi kiya (breakpoint_threshold_amount)
        self.chunker = SemanticChunker(
            self.embedding, 
            breakpoint_threshold_type="percentile", 
            breakpoint_threshold_amount=85 
        )

    def create_chunks(self) -> List[Dict]:
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return []

        with open(self.input_file, "r", encoding="utf-8") as f:
            documents = json.load(f)

        logger.info(f"Splitting {len(documents)} docs into semantic chunks. This take a few minutes...")
        semantic_chunks = []
        
        for doc in documents:
            text = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
            
            try:
                text_chunks = self.chunker.split_text(text)
                for i, t in enumerate(text_chunks):
                    semantic_chunks.append({
                        "content": t,
                        "metadata": {**metadata, "chunk_index": i, "strategy": "semantic"}
                    })
            except Exception as e:
                logger.warning(f"Failed to split: {str(e)}")

        self._save(semantic_chunks)
        return semantic_chunks

    def _save(self, chunks: List[Dict]) -> None:
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        logger.info(f"Success! Semantic chunks saved successfully at: {self.output_file}")

if __name__ == "__main__":
    chunker = FinanceSemanticChunker()
    chunker.create_chunks()
