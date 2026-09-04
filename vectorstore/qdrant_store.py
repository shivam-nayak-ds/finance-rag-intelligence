"""Qdrant vector store integration supporting local and remote instances."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from config.settings import (
    ALL_QDRANT_COLLECTIONS,
    EMBEDDING_DIMENSION,
    QDRANT_API_KEY,
    QDRANT_PERSIST_DIR,
    QDRANT_URL,
)
from models.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Generates a deterministic RFC 4122 UUID from a string chunk identifier."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantStore:
    """Vector database client for Qdrant collections."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.url = url or QDRANT_URL
        self.api_key = api_key or QDRANT_API_KEY
        self.persist_dir = persist_dir or QDRANT_PERSIST_DIR

        if self.url:
            logger.info(f"Connecting to remote Qdrant instance: {self.url}")
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Using local embedded Qdrant store at: {self.persist_dir}")
            self.client = QdrantClient(path=self.persist_dir)

        self.init_collections()

    def create_collection_if_not_exists(
        self,
        collection_name: str,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        """Creates collection with Cosine distance metric and payload indexes if not present."""
        if not self.client.collection_exists(collection_name):
            logger.info(f"Creating Qdrant collection '{collection_name}' (dim={dimension}, metric=cosine)")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

            if self.url:
                payload_indexes = [
                    ("unit", PayloadSchemaType.INTEGER),
                    ("topic", PayloadSchemaType.KEYWORD),
                    ("source_type", PayloadSchemaType.KEYWORD),
                    ("year", PayloadSchemaType.INTEGER),
                ]
                for field_name, field_schema in payload_indexes:
                    try:
                        self.client.create_payload_index(
                            collection_name=collection_name,
                            field_name=field_name,
                            field_schema=field_schema,
                        )
                    except Exception as err:
                        logger.debug(f"Payload index creation error on '{collection_name}.{field_name}': {err}")

    def init_collections(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        """Initializes default collections."""
        for coll in ALL_QDRANT_COLLECTIONS:
            self.create_collection_if_not_exists(coll, dimension=dimension)

    def upsert_documents(
        self,
        collection_name: str,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        batch_size: int = 100,
    ) -> int:
        """Batch upserts documents and their corresponding vector embeddings."""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Document/Embedding length mismatch: {len(documents)} docs vs {len(embeddings)} vectors"
            )

        if not documents:
            return 0

        self.create_collection_if_not_exists(collection_name)
        total_points = len(documents)

        points: List[PointStruct] = []
        for doc, emb in zip(documents, embeddings):
            payload = doc.model_dump()
            point_id = chunk_id_to_uuid(doc.chunk_id)
            points.append(PointStruct(id=point_id, vector=list(emb), payload=payload))

        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)

        logger.info(f"Upserted {total_points} vectors into collection '{collection_name}'")
        return total_points

    def similarity_search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        top_k: int = 10,
        unit: Optional[int] = None,
        topic: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Document]:
        """Performs cosine vector search with optional payload filters."""
        filter_conditions: List[FieldCondition] = []
        if unit is not None:
            filter_conditions.append(FieldCondition(key="unit", match=MatchValue(value=unit)))
        if topic is not None:
            filter_conditions.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
        if source_type is not None:
            filter_conditions.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        query_response = self.client.query_points(
            collection_name=collection_name,
            query=list(query_vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results: List[Document] = []
        for scored_point in query_response.points:
            payload = scored_point.payload or {}
            try:
                doc = Document.model_validate(payload)
                doc.dense_score = float(scored_point.score)
                results.append(doc)
            except Exception as parse_err:
                logger.warning(f"Error parsing point payload {scored_point.id}: {parse_err}")

        return results

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Returns metadata and count statistics for a given collection."""
        if not self.client.collection_exists(collection_name):
            return {"exists": False, "points_count": 0}

        info = self.client.get_collection(collection_name=collection_name)
        return {
            "exists": True,
            "points_count": getattr(info, "points_count", 0) or getattr(info, "vectors_count", 0) or 0,
            "status": str(info.status),
        }

    def close(self) -> None:
        """Closes the client connection."""
        try:
            self.client.close()
        except Exception:
            pass
