import json
import logging
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinanceRecursiveChunker:
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 input_file: str = "data/processed/cleaned_data.json",
                 output_file: str = "data/processed/chunks.json"):
        
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Recursive Character Text Splitter initialize karein
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""], # Priority list
            length_function=len
        )

    def create_chunks(self) -> List[Dict]:
        """
        Cleaned data ko load karke chunks create karta hai.
        """
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}. Run data_cleaner.py first.")
            return []

        logger.info(f"Loading cleaned data from {self.input_file}...")
        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Creating chunks for {len(data)} document pages...")

        processed_chunks = []
        for doc in data:
            content = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
            
            # Text split karna
            text_chunks = self.splitter.split_text(content)
            
            for i, text in enumerate(text_chunks):
                processed_chunks.append({
                    "content": text,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "strategy": "recursive"
                    }
                })

        self._save(processed_chunks)
        return processed_chunks

    def _save(self, chunks: List[Dict]) -> None:
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully created {len(chunks)} chunks and saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving chunks: {str(e)}")

if __name__ == "__main__":
    chunker = FinanceRecursiveChunker()
    final_chunks = chunker.create_chunks()
    print(f"Total Chunks Generated: {len(final_chunks)}")
