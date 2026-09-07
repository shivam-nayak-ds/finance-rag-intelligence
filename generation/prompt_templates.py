"""Prompt templates for SyllAIq CS tutor persona."""

from typing import List

from models.documents import Citation, Document, Intent


# ──────────────────────────────────────────────────────────────────
# System Prompt — Tutor Persona
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are SyllAIq — an expert AI tutor for RGPV CSE students preparing for exams.

Your persona:
- You explain concepts clearly in simple English (mixing Hinglish is OK when helpful)
- You always cite your sources using [Source N] notation
- You highlight exam-important points with ⭐
- You never fabricate information — if unsure, say "Mujhe iske baare mein pakka nahi pata"
- You keep answers concise and exam-focused

Citation format example:
  "Deadlock ke 4 necessary conditions hain: mutual exclusion, hold and wait,
   no preemption, and circular wait [Source 1]."

Always end concept explanations with:
  "📚 Sources: [list used sources]"
"""


# ──────────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────────

def _format_document(doc: Document, index: int) -> str:
    """Format a single retrieved chunk with its citation label."""
    source_label = f"[Source {index}]"

    if doc.source_type == "textbook":
        header = f"{source_label} — {doc.book or 'OS Textbook'}"
        if doc.chapter:
            header += f", Chapter {doc.chapter}"
        if doc.page_start:
            header += f", Page {doc.page_start}"
    elif doc.source_type == "pyq":
        header = f"{source_label} — RGPV PYQ {doc.year or ''}"
        if doc.marks:
            header += f" ({doc.marks} marks)"
    elif doc.source_type == "syllabus":
        header = f"{source_label} — RGPV OS Syllabus"
        if doc.unit:
            header += f", Unit {doc.unit}"
        if doc.topic:
            header += f": {doc.topic}"
    else:
        header = source_label

    return f"{header}\n{doc.text.strip()}"


def build_context_block(documents: List[Document]) -> str:
    """Combine retrieved documents into a numbered context block."""
    if not documents:
        return "No relevant context found in knowledge base."
    parts = [_format_document(doc, i + 1) for i, doc in enumerate(documents)]
    return "\n\n---\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────
# Intent-specific prompt builders
# ──────────────────────────────────────────────────────────────────

def build_concept_prompt(query: str, context: str) -> str:
    return f"""Using ONLY the sources below, answer the student's question clearly and concisely.
Mark exam-important points with ⭐. Use [Source N] citations.

SOURCES:
{context}

STUDENT QUESTION: {query}

ANSWER:"""


def build_pyq_prompt(query: str, context: str) -> str:
    return f"""The student wants to see Previous Year Questions (PYQs).
List all relevant PYQs from the sources below with year, marks, and brief answer hints.

SOURCES:
{context}

STUDENT QUERY: {query}

PYQ LIST:"""


def build_importance_prompt(query: str, context: str) -> str:
    return f"""Based on PYQ frequency in the sources, rank topics by exam importance.
Show year-wise appearance count for each topic.

SOURCES:
{context}

STUDENT QUERY: {query}

TOPIC IMPORTANCE ANALYSIS:"""


def build_syllabus_prompt(query: str, context: str) -> str:
    return f"""Answer the student's syllabus query using ONLY the official RGPV syllabus below.
Be specific about units and topics.

SOURCES:
{context}

STUDENT QUERY: {query}

SYLLABUS ANSWER:"""


def build_prompt(query: str, context: str, intent: str) -> str:
    """Select appropriate prompt template based on classified intent."""
    intent_map = {
        Intent.CONCEPT_EXPLANATION: build_concept_prompt,
        Intent.PYQ_RETRIEVAL: build_pyq_prompt,
        Intent.TOPIC_IMPORTANCE: build_importance_prompt,
        Intent.SYLLABUS_LOOKUP: build_syllabus_prompt,
    }
    builder = intent_map.get(intent, build_concept_prompt)
    return builder(query, context)
