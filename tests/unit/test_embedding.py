"""Unit tests for HFEmbedder."""

import math
import pytest
from embedding.hf_embedder import HFEmbedder


@pytest.fixture(scope="module")
def embedder():
    return HFEmbedder()


def test_embedder_dimension(embedder):
    assert embedder.dimension == 384


def test_embed_query(embedder):
    query = "What are the four necessary conditions for deadlock?"
    vector = embedder.embed_query(query)

    assert isinstance(vector, list)
    assert len(vector) == 384

    norm = math.sqrt(sum(x * x for x in vector))
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_documents_batch(embedder):
    chunks = [
        "Operating System manages hardware resources.",
        "Banker's algorithm is used for deadlock avoidance.",
        "Virtual memory uses demand paging and page tables.",
    ]
    vectors = embedder.embed_documents(chunks)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 384
        norm = math.sqrt(sum(x * x for x in v))
        assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_empty_query(embedder):
    vec = embedder.embed_query("")
    assert len(vec) == 384
    assert all(x == 0.0 for x in vec)
