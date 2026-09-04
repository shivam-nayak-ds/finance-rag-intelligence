"""Unit tests for QdrantStore."""

import uuid
import pytest

from models.documents import Document, SourceType
from vectorstore.qdrant_store import QdrantStore, chunk_id_to_uuid


@pytest.fixture
def temp_qdrant(tmp_path):
    store = QdrantStore(persist_dir=str(tmp_path / "qdrant_test"))
    return store


def test_deterministic_uuid():
    chunk_id = "galvin_os_p318_c7"
    uid1 = chunk_id_to_uuid(chunk_id)
    uid2 = chunk_id_to_uuid(chunk_id)

    assert uid1 == uid2
    assert str(uuid.UUID(uid1)) == uid1


def test_collections_initialization(temp_qdrant):
    for coll in ["os_textbook", "os_pyqs", "os_syllabus"]:
        stats = temp_qdrant.get_collection_stats(coll)
        assert stats["exists"] is True


def test_upsert_and_similarity_search(temp_qdrant):
    coll = "os_textbook"

    doc1 = Document(
        chunk_id="test_doc_1",
        text="Deadlock occurs when mutual exclusion, hold and wait, no preemption, circular wait hold.",
        source_type=SourceType.TEXTBOOK,
        unit=4,
        topic="deadlock",
        book="Galvin OS 10th Ed",
        chapter=7,
        page_start=318,
    )
    doc2 = Document(
        chunk_id="test_doc_2",
        text="CPU scheduling algorithms include FCFS, SJF, and Round Robin.",
        source_type=SourceType.TEXTBOOK,
        unit=3,
        topic="scheduling",
        book="Galvin OS 10th Ed",
        chapter=5,
        page_start=200,
    )

    vec1 = [0.0] * 384
    vec1[0] = 1.0

    vec2 = [0.0] * 384
    vec2[1] = 1.0

    temp_qdrant.upsert_documents(coll, [doc1, doc2], [vec1, vec2])

    stats = temp_qdrant.get_collection_stats(coll)
    assert stats["points_count"] == 2

    results = temp_qdrant.similarity_search(coll, query_vector=vec1, top_k=2)
    assert len(results) >= 1
    top_result = results[0]
    assert top_result.chunk_id == "test_doc_1"
    assert top_result.unit == 4
    assert top_result.dense_score is not None
    assert top_result.dense_score > 0.9


def test_payload_unit_filtering(temp_qdrant):
    coll = "os_textbook"

    doc_u4 = Document(
        chunk_id="doc_u4",
        text="Banker's algorithm ensures system safety.",
        source_type=SourceType.TEXTBOOK,
        unit=4,
        topic="deadlock",
    )
    doc_u3 = Document(
        chunk_id="doc_u3",
        text="Demand paging uses page replacement algorithms.",
        source_type=SourceType.TEXTBOOK,
        unit=3,
        topic="paging",
    )

    vec_same = [0.1] * 384
    temp_qdrant.upsert_documents(coll, [doc_u4, doc_u3], [vec_same, vec_same])

    results_u3 = temp_qdrant.similarity_search(coll, query_vector=vec_same, top_k=5, unit=3)
    assert len(results_u3) == 1
    assert results_u3[0].chunk_id == "doc_u3"
    assert results_u3[0].unit == 3

    results_u4 = temp_qdrant.similarity_search(coll, query_vector=vec_same, top_k=5, unit=4)
    assert len(results_u4) == 1
    assert results_u4[0].chunk_id == "doc_u4"
    assert results_u4[0].unit == 4
