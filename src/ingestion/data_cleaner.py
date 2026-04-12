import re
import json
import logging
from pathlib import Path
from typing import List, Dict

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinanceDataCleaner:
    def __init__(self, input_file: str = "data/processed/processed_data.json", output_file: str = "data/processed/cleaned_data.json"):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """
        Finance specific cleaning logic.
        """
        if not text:
            return ""

        # 1. Newlines and multiple spaces fix
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)

        # 2. Fix hyphenated words at line breaks (e.g., "in-vestment" -> "investment")
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

        # 3. Remove special characters that aren't useful for RAG 
        # (Keeping currency symbols and decimal points)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII

        # 4. Strip leading/trailing whitespaces
        text = text.strip()

        return text

    def process(self) -> List[Dict]:
        """
        Loads the raw JSON, cleans it, and saves the cleaned version.
        """
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}. Run pdf_loader.py first.")
            return []

        logger.info(f"Loading data from {self.input_file}...")
        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Cleaning {len(data)} document pages...")
        cleaned_data = []
        for doc in data:
            cleaned_content = self.clean_text(doc.get("page_content", ""))
            
            # Skip very short content (likely noise or empty pages)
            if len(cleaned_content) < 50:
                continue
                
            cleaned_data.append({
                "page_content": cleaned_content,
                "metadata": doc.get("metadata", {})
            })

        self._save(cleaned_data)
        return cleaned_data

    def _save(self, data: List[Dict]) -> None:
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Cleaned data saved successfully at: {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving cleaned data: {str(e)}")

if __name__ == "__main__":
    cleaner = FinanceDataCleaner()
    cleaned_docs = cleaner.process()
    print(f"Total pages after cleaning: {len(cleaned_docs)}")
