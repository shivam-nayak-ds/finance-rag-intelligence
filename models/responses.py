"""
SyllAIq — Pydantic Response Models
====================================
Defines the structured response objects for the RAG pipeline
and FastAPI endpoints.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from models.documents import Citation, ConfidenceLevel, Intent


# ─────────────────────────────────────────────────────────────
# Internal Pipeline Result
# ─────────────────────────────────────────────────────────────

class RAGResult(BaseModel):
    """
    Internal result object returned by the LangGraph pipeline.
    Carries everything needed to build the final API response.
    """

    # Core answer
    answer    : str = Field(..., description="The generated answer text")
    citations : list[Citation] = Field(default_factory=list, description="Source citations")

    # Quality signals
    confidence      : ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    grounding_score : float = Field(default=0.0, description="NLI-based groundedness (0–1)")
    intent          : Intent = Field(default=Intent.UNKNOWN)

    # Self-RAG signals
    self_corrected        : bool = Field(default=False, description="Was self-correction triggered?")
    self_correction_count : int  = Field(default=0, description="Number of self-correction attempts")
    retrieval_failed      : bool = Field(default=False, description="Was retrieval insufficient?")
    web_search_used       : bool = Field(default=False, description="Was web search triggered?")

    # Performance
    retrieval_time_ms  : float = Field(default=0.0)
    reranking_time_ms  : float = Field(default=0.0)
    generation_time_ms : float = Field(default=0.0)
    total_latency_ms   : float = Field(default=0.0)
    total_tokens       : int   = Field(default=0)

    # Optional warning for low confidence answers
    warning : Optional[str] = Field(None, description="Warning message for low confidence answers")

    # Error info
    error  : Optional[str] = Field(None)
    status : str = Field(default="success")  # success | partial | failed


# ─────────────────────────────────────────────────────────────
# FastAPI Request & Response Schemas
# ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request body for POST /api/v1/ask"""
    query      : str = Field(..., min_length=1, max_length=500, description="Student's question")
    subject    : str = Field(default="Operating Systems", description="Subject filter")
    session_id : Optional[str] = Field(None, description="Session ID for conversation memory")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Deadlock ke 4 necessary conditions kya hain?",
                "subject": "Operating Systems",
                "session_id": "student-abc-123"
            }
        }


class AskResponse(BaseModel):
    """Response body for POST /api/v1/ask"""
    answer     : str
    citations  : list[Citation]
    confidence : str                    # "high" | "medium" | "low"
    intent     : str                    # "concept" | "pyq" | "importance" | "syllabus"
    warning    : Optional[str] = None   # Shown when confidence is low
    self_corrected    : bool = False
    web_search_used   : bool = False
    latency_ms        : float = 0.0
    request_id        : str = ""        # For tracing

    @classmethod
    def from_rag_result(cls, result: RAGResult, request_id: str = "") -> "AskResponse":
        """Build API response from internal pipeline result."""
        return cls(
            answer=result.answer,
            citations=result.citations,
            confidence=result.confidence,
            intent=result.intent,
            warning=result.warning,
            self_corrected=result.self_corrected,
            web_search_used=result.web_search_used,
            latency_ms=result.total_latency_ms,
            request_id=request_id,
        )


class HealthResponse(BaseModel):
    """Response for GET /health"""
    status  : str = "ok"
    version : str = "0.1.0"
    project : str = "SyllAIq"
