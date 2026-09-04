"""Vector index construction script for Qdrant collections."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    CHUNKS_JSON,
    QDRANT_COLLECTION_PYQS,
    QDRANT_COLLECTION_SYLLABUS,
    QDRANT_COLLECTION_TEXTBOOK,
)
from embedding.hf_embedder import HFEmbedder
from models.documents import Document, SourceType
from utils.logger import get_logger
from vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


def build_vector_index(
    chunks_path: Path = CHUNKS_JSON,
    sample_size: int = 0,
    batch_size: int = 64,
) -> Dict[str, int]:
    """Generates embeddings for processed chunks and stores them in Qdrant."""
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at '{chunks_path}'. Run 'scripts/run_chunking.py' first."
        )

    start_time = time.perf_counter()
    logger.info("Starting vector index build from: %s", chunks_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    if not isinstance(raw_chunks, list):
        raise ValueError(f"Expected a list of chunks, got {type(raw_chunks)}")

    if sample_size > 0:
        raw_chunks = raw_chunks[:sample_size]
        logger.info("Using sample size of %d chunks", sample_size)

    total_chunks = len(raw_chunks)
    logger.info("Loaded %d total chunks", total_chunks)

    buckets: Dict[str, List[Document]] = {
        QDRANT_COLLECTION_TEXTBOOK: [],
        QDRANT_COLLECTION_PYQS: [],
        QDRANT_COLLECTION_SYLLABUS: [],
    }

    for item in raw_chunks:
        try:
            doc = Document.model_validate(item)
            if doc.source_type == SourceType.TEXTBOOK:
                buckets[QDRANT_COLLECTION_TEXTBOOK].append(doc)
            elif doc.source_type == SourceType.PYQ:
                buckets[QDRANT_COLLECTION_PYQS].append(doc)
            elif doc.source_type == SourceType.SYLLABUS:
                buckets[QDRANT_COLLECTION_SYLLABUS].append(doc)
            else:
                buckets[QDRANT_COLLECTION_TEXTBOOK].append(doc)
        except Exception as parse_err:
            logger.warning("Skipping malformed chunk: %s", parse_err)

    logger.info(
        "Partitioned: %d textbook chunks, %d PYQs, %d syllabus units",
        len(buckets[QDRANT_COLLECTION_TEXTBOOK]),
        len(buckets[QDRANT_COLLECTION_PYQS]),
        len(buckets[QDRANT_COLLECTION_SYLLABUS]),
    )

    embedder = HFEmbedder(batch_size=batch_size)
    qdrant = QdrantStore()

    indexed_counts: Dict[str, int] = {}

    for collection_name, docs in buckets.items():
        if not docs:
            logger.info("Collection '%s' is empty, skipping", collection_name)
            indexed_counts[collection_name] = 0
            continue

        total_coll_docs = len(docs)
        chunk_step = 500
        total_upserted = 0
        coll_start = time.perf_counter()

        logger.info("Indexing %d items into collection '%s'", total_coll_docs, collection_name)

        for start_idx in range(0, total_coll_docs, chunk_step):
            end_idx = min(start_idx + chunk_step, total_coll_docs)
            batch_docs = docs[start_idx:end_idx]
            batch_texts = [d.text for d in batch_docs]

            batch_embeddings = embedder.embed_documents(batch_texts)
            qdrant.upsert_documents(
                collection_name=collection_name,
                documents=batch_docs,
                embeddings=batch_embeddings,
                batch_size=100,
            )
            total_upserted += len(batch_docs)
            elapsed = time.perf_counter() - coll_start
            pct = (total_upserted / total_coll_docs) * 100
            rate = total_upserted / max(elapsed, 0.001)
            logger.info(
                "[%s] Progress: %d/%d (%.1f%%) | %.1f items/sec",
                collection_name,
                total_upserted,
                total_coll_docs,
                pct,
                rate,
            )

        indexed_counts[collection_name] = total_upserted

    total_duration = time.perf_counter() - start_time
    logger.info("Vector index build completed in %.2f seconds", total_duration)
    for coll, count in indexed_counts.items():
        stats = qdrant.get_collection_stats(coll)
        logger.info("Collection '%s': %d vectors active", coll, stats.get("points_count", count))

    return indexed_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qdrant vector index for SyllAIq")
    parser.add_argument("--sample", type=int, default=0, help="Number of chunks to index (0 = all)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding")
    args = parser.parse_args()

    build_vector_index(sample_size=args.sample, batch_size=args.batch_size)
