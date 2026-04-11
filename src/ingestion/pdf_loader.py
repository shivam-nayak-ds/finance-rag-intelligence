import json
import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancePDFLoader:
    def __init__(self, data_path: str = "data/raw", processed_path: str = "data/processed"):
        self.data_path = Path(data_path)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_and_save(self) -> List[Document]:
        documents = []
        # Saari PDF files load karna
        pdf_files = list(self.data_path.glob("*.pdf"))
        
        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf))
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading {pdf}: {str(e)}")
            
        self._save_to_json(documents)
        return documents

    def _save_to_json(self, documents: List[Document]) -> None:
        save_file = self.processed_path / "processed_data.json"
        
        try:
            # SHI KIYA: doc variable define kiya aur metadata se quotes hataye
            processed_data = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in documents
            ]
            
            # SHI KIYA: File handle 'f' define kiya aur alignment theek ki
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Data verification file created at: {save_file}")
            
        except Exception as e:
            logger.warning(f"Could not save JSON backup: {str(e)}")

## Unit test block 

if __name__ == "__main__":
    # SHI KIYA: if__name__ typos theek kiye
    loader = FinancePDFLoader()
    documents = loader.load_and_save()
    print(f"Successfully loaded {len(documents)} pages")
