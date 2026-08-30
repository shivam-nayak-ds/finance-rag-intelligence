"""
SyllAIq — Core Pydantic Models: Documents & Citations
=======================================================
Defines the core data structures used throughout the pipeline.
Every retrieved chunk is a Document. Every source reference is a Citation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    """Type of source document."""
    TEXTBOOK = "textbook"
    PYQ      = "pyq"
    SYLLABUS = "syllabus"
    WEB      = "web"


class ConfidenceLevel(str, Enum):
    """Answer confidence level based on retrieval + grounding scores."""
    HIGH   = "high"    # >= 0.85
    MEDIUM = "medium"  # >= 0.60
    LOW    = "low"     # <  0.60


class Intent(str, Enum):
    """Classified intent of the user's query."""
    CONCEPT_EXPLANATION = "concept"
    PYQ_RETRIEVAL       = "pyq"
    TOPIC_IMPORTANCE    = "importance"
    SYLLABUS_LOOKUP     = "syllabus"
    UNKNOWN             = "unknown"


# ─────────────────────────────────────────────────────────────
# Document — Core retrieval unit
# ─────────────────────────────────────────────────────────────

class Document(BaseModel):
    """
    A single chunk of retrieved content with full source metadata.

    Every chunk stored in ChromaDB must have all required fields.
    This is the standard object passed between all pipeline nodes.
    """

    # Content
    chunk_id   : str = Field(..., description="Unique chunk identifier")
    text       : str = Field(..., description="The actual chunk text content")

    # Source classification
    source_type: SourceType = Field(..., description="Type of source: textbook/pyq/syllabus")
    subject    : str = Field(default="Operating Systems", description="Academic subject")
    university : str = Field(default="RGPV", description="University name")

    # Textbook-specific metadata
    book       : Optional[str] = Field(None, description="Book title, e.g. 'Galvin OS 10th Ed'")
    chapter    : Optional[int] = Field(None, description="Chapter number")
    page_start : Optional[int] = Field(None, description="Starting page number")
    page_end   : Optional[int] = Field(None, description="Ending page number")
    unit       : Optional[int] = Field(None, description="RGPV syllabus unit number (1–5)")

    # PYQ-specific metadata
    year       : Optional[int]  = Field(None, description="Exam year, e.g. 2023")
    semester   : Optional[int]  = Field(None, description="Semester number, e.g. 5")
    marks      : Optional[int]  = Field(None, description="Question marks, e.g. 7")
    question_no: Optional[str]  = Field(None, description="Question number on paper")

    # Common metadata
    topic      : Optional[str]  = Field(None, description="Main topic of this chunk")
    chunk_index: Optional[int]  = Field(None, description="Index of chunk within parent doc")
    char_count : Optional[int]  = Field(None, description="Character count of text")

    # Retrieval scores (populated during retrieval)
    dense_score   : Optional[float] = Field(None, description="ChromaDB cosine similarity score")
    bm25_score    : Optional[float] = Field(None, description="BM25 retrieval score")
    rrf_score     : Optional[float] = Field(None, description="RRF fusion score")
    reranker_score: Optional[float] = Field(None, description="Cohere reranker relevance score")
    nli_score     : Optional[float] = Field(None, description="NLI relevance grading score")

    class Config:
        use_enum_values = True


# ─────────────────────────────────────────────────────────────
# Citation — Source reference in the final answer
# ─────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """
    A verifiable source reference attached to the generated answer.

    Citations must map 1:1 with actual retrieved Documents.
    Fabricated citations are caught by verify_citations node.
    """

    source_number : int        = Field(..., description="Citation number in answer: [Source 1]")
    chunk_id      : str        = Field(..., description="The chunk_id of the source Document")
    source_type   : SourceType = Field(..., description="Type of source")
    display_text  : str        = Field(..., description="Human-readable citation label")

    # For textbook citations
    book          : Optional[str] = None
    chapter       : Optional[int] = None
    page          : Optional[int] = None

    # For PYQ citations
    year          : Optional[int] = None
    marks         : Optional[int] = None

    # For web citations (Phase 11)
    url           : Optional[str] = None
    trust_tier    : Optional[int] = None

    @classmethod
    def from_document(cls, doc: Document, source_number: int) -> "Citation":
        """Build a Citation from a retrieved Document."""
        if doc.source_type == SourceType.TEXTBOOK:
            display = f"{doc.book or 'Textbook'}"
            if doc.chapter:
                display += f", Ch.{doc.chapter}"
            if doc.page_start:
                display += f", Page {doc.page_start}"
        elif doc.source_type == SourceType.PYQ:
            display = f"RGPV OS PYQ {doc.year or ''}"
            if doc.marks:
                display += f" ({doc.marks} marks)"
        elif doc.source_type == SourceType.SYLLABUS:
            display = f"RGPV OS Syllabus"
            if doc.unit:
                display += f", Unit {doc.unit}"
        else:
            display = doc.topic or "Unknown Source"

        return cls(
            source_number=source_number,
            chunk_id=doc.chunk_id,
            source_type=doc.source_type,
            display_text=display,
            book=doc.book,
            chapter=doc.chapter,
            page=doc.page_start,
            year=doc.year,
            marks=doc.marks,
        )

    class Config:
        use_enum_values = True
