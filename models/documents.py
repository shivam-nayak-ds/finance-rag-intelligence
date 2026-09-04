"""Domain models for documents, citations, and pipeline data transfer objects."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Source classification for documents."""
    TEXTBOOK = "textbook"
    PYQ = "pyq"
    SYLLABUS = "syllabus"
    WEB = "web"


class ConfidenceLevel(str, Enum):
    """Confidence level assigned to synthesized responses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Intent(str, Enum):
    """Classified user query intent."""
    CONCEPT_EXPLANATION = "concept"
    PYQ_RETRIEVAL = "pyq"
    TOPIC_IMPORTANCE = "importance"
    SYLLABUS_LOOKUP = "syllabus"
    UNKNOWN = "unknown"


class Document(BaseModel):
    """Canonical document chunk representation passed through pipeline stages."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    source_type: SourceType = Field(..., description="Source classification")
    subject: str = Field(default="Operating Systems", description="Academic subject")
    university: str = Field(default="RGPV", description="University name")

    # Textbook metadata
    book: Optional[str] = Field(None, description="Book title")
    chapter: Optional[int] = Field(None, description="Chapter number")
    page_start: Optional[int] = Field(None, description="Starting page number")
    page_end: Optional[int] = Field(None, description="Ending page number")
    unit: Optional[int] = Field(None, description="Syllabus unit number (1-5)")

    # PYQ metadata
    year: Optional[int] = Field(None, description="Exam year")
    semester: Optional[int] = Field(None, description="Semester number")
    marks: Optional[int] = Field(None, description="Question marks")
    question_no: Optional[str] = Field(None, description="Question number")

    # Common metadata
    topic: Optional[str] = Field(None, description="Topic title")
    chunk_index: Optional[int] = Field(None, description="Chunk index within document")
    char_count: Optional[int] = Field(None, description="Character count")

    # Retrieval scores
    dense_score: Optional[float] = Field(None, description="Dense vector cosine similarity score")
    bm25_score: Optional[float] = Field(None, description="BM25 sparse score")
    rrf_score: Optional[float] = Field(None, description="RRF score")
    reranker_score: Optional[float] = Field(None, description="Reranker score")
    nli_score: Optional[float] = Field(None, description="NLI relevance score")

    model_config = {"use_enum_values": True}


class Citation(BaseModel):
    """Citation reference mapped to a retrieved source document."""

    source_number: int = Field(..., description="Citation index")
    chunk_id: str = Field(..., description="Target document chunk_id")
    source_type: SourceType = Field(..., description="Source classification")
    display_text: str = Field(..., description="Human-readable citation text")

    # Textbook citation fields
    book: Optional[str] = None
    chapter: Optional[int] = None
    page: Optional[int] = None

    # PYQ citation fields
    year: Optional[int] = None
    marks: Optional[int] = None

    # Web citation fields
    url: Optional[str] = None
    trust_tier: Optional[int] = None

    @classmethod
    def from_document(cls, doc: Document, source_number: int) -> "Citation":
        """Constructs Citation from a retrieved Document."""
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
            display = "RGPV OS Syllabus"
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

    model_config = {"use_enum_values": True}
