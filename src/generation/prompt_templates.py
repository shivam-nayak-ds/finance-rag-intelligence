"""
Prompt Templates — Finance RAG Intelligence
Carefully crafted prompts to ground LLM responses in retrieved context.
"""

# ─────────────────────────────────────────────
# System Prompt: Sets the LLM's persona & rules
# ─────────────────────────────────────────────
FINANCE_SYSTEM_PROMPT = """You are FinanceRAG, an expert financial analyst assistant.
You answer questions STRICTLY based on the context provided below.

Rules you MUST follow:
1. Only use information from the provided context chunks.
2. If the answer is not in the context, say: "I don't have enough information in the provided documents to answer this."
3. Be precise with numbers — do not round or approximate financial figures.
4. Cite which context chunk your answer came from (e.g., [Source 1], [Source 2]).
5. Keep your answer concise, structured, and professional.
"""

# ─────────────────────────────────────────────
# User Prompt: Injects query + retrieved chunks
# ─────────────────────────────────────────────
def build_rag_prompt(query: str, context_chunks: list[str]) -> str:
    """
    Builds the final prompt by injecting retrieved chunks as numbered context.

    Args:
        query: The user's financial question.
        context_chunks: List of re-ranked document chunks.

    Returns:
        A formatted prompt string ready for the LLM.
    """
    if not context_chunks:
        context_block = "No context available."
    else:
        context_block = "\n\n".join(
            f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
        )

    prompt = f"""CONTEXT:
{context_block}

─────────────────────────────────────────────
QUESTION: {query}
─────────────────────────────────────────────
ANSWER (based only on the context above):"""

    return prompt
