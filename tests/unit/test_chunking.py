"""Unit tests for document chunking strategies."""

import pytest
from chunking.pyq_chunker import PYQChunker
from chunking.recursive_chunker import CSRecursiveChunker
from chunking.syllabus_chunker import SyllabusChunker
from models.documents import Document, SourceType


class TestCSRecursiveChunker:
    """Tests for recursive textbook chunker."""

    def test_chunking_preserves_parent_metadata(self):
        parent_doc = Document(
            chunk_id="galvin_p318",
            text=(
                "A deadlock is a situation where a set of processes are blocked because "
                "each process is holding a resource and waiting for another resource. "
                "Four conditions must hold: Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait. "
            ) * 10,  # ~1400 chars, will split into >=2 chunks
            source_type=SourceType.TEXTBOOK,
            book="Operating System Concepts (Galvin 10th Ed)",
            chapter=7,
            page_start=318,
            page_end=318,
            unit=4,
            topic="Deadlock",
        )

        chunker = CSRecursiveChunker(chunk_size=800, chunk_overlap=150)
        chunks = chunker.split_document(parent_doc)

        assert len(chunks) >= 2
        for idx, chunk in enumerate(chunks):
            assert chunk.book == "Operating System Concepts (Galvin 10th Ed)"
            assert chunk.chapter == 7
            assert chunk.page_start == 318
            assert chunk.unit == 4
            assert chunk.topic == "Deadlock"
            assert chunk.chunk_id == f"galvin_p318_c{idx}"
            assert chunk.chunk_index == idx

    def test_short_document_not_split(self):
        doc = Document(
            chunk_id="short_doc",
            text="This is a brief 50 character textbook note.",
            source_type=SourceType.TEXTBOOK,
            page_start=10,
            unit=1,
            topic="Introduction",
        )
        chunker = CSRecursiveChunker(chunk_size=800, chunk_overlap=150)
        chunks = chunker.split_document(doc)

        assert len(chunks) == 1
        assert chunks[0].text == doc.text
        assert chunks[0].chunk_index == 0


class TestPYQChunker:
    """Tests for atomic PYQ chunker."""

    def test_pyq_never_split(self):
        pyq_doc = Document(
            chunk_id="RGPV_2023_q5b",
            text="Question 5.b (7 Marks) [RGPV 2023]: What do you mean by deadlock prevention? Explain tape drives numerical.",
            source_type=SourceType.PYQ,
            year=2023,
            marks=7,
            unit=4,
            topic="Deadlock Prevention",
        )

        chunker = PYQChunker()
        chunks = chunker.split_document(pyq_doc)

        assert len(chunks) == 1
        assert chunks[0].source_type == SourceType.PYQ
        assert chunks[0].year == 2023
        assert chunks[0].marks == 7
        assert chunks[0].unit == 4


class TestSyllabusChunker:
    """Tests for syllabus chunker."""

    def test_syllabus_unit_preserved(self):
        s_doc = Document(
            chunk_id="syllabus_u4",
            text="RGPV OS Syllabus — Unit 4: I/O & Deadlocks\n- Concurrency\n- Semaphores\n- Deadlocks",
            source_type=SourceType.SYLLABUS,
            unit=4,
            topic="Deadlocks",
        )

        chunker = SyllabusChunker()
        chunks = chunker.split_document(s_doc)

        assert len(chunks) == 1
        assert chunks[0].source_type == SourceType.SYLLABUS
        assert chunks[0].unit == 4
