"""Sentence embedding utility using HuggingFace models."""

from typing import Final, List, Optional, Sequence
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class HFEmbedder:
    """Wrapper around SentenceTransformer for generating dense vector embeddings."""

    _BGE_QUERY_PREFIX: Final[str] = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name or EMBEDDING_MODEL
        self.batch_size = batch_size
        self._is_bge = "bge" in self.model_name.lower()

        logger.info(f"Loading embedding model '{self.model_name}' (device: {device or 'auto'})")
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
        except Exception as err:
            logger.warning(f"Failed to fetch model from hub ({err}), attempting local cache load")
            self.model = SentenceTransformer(self.model_name, device=device, local_files_only=True)

        get_dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self._dimension = int(get_dim_fn())

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimensionality."""
        return self._dimension

    def embed_query(self, query: str) -> List[float]:
        """Embeds a search query with task instruction if applicable."""
        if not query or not query.strip():
            logger.warning("Empty query provided to embed_query; returning zero vector")
            return [0.0] * self._dimension

        text = f"{self._BGE_QUERY_PREFIX}{query.strip()}" if self._is_bge else query.strip()
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embeds a collection of document texts in batches."""
        if not texts:
            return []

        cleaned_texts = [t if (t and t.strip()) else " " for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
