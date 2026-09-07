"""LLM-based query rewriter to improve retrieval quality."""

import os
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_REWRITE_PROMPT = """\
You are a search query optimizer for an Operating Systems exam preparation system.

Your job: Convert the student's question into an optimal search query that will \
retrieve the most relevant textbook sections, PYQs, and syllabus content.

Rules:
- Expand abbreviations (e.g. "OS" → "Operating Systems", "DB" → "Database")
- Add relevant technical keywords the student might have missed
- If the query is in Hindi/Hinglish, translate key terms to English
- Keep the rewritten query concise (max 2 sentences)
- Do NOT add anything that changes the meaning
- Return ONLY the rewritten query, no explanation

Student question: {query}
Rewritten search query:"""

_REWRITE_PROMPT_WITH_CONTEXT = """\
You are a search query optimizer for an Operating Systems exam preparation system.

Recent conversation context:
{context}

Student question: {query}

Your job: Convert the student's follow-up question into an optimal standalone search query.
- Resolve references like "iska", "this", "that", "it", "previous" using the context.
- Add relevant technical keywords for Operating Systems retrieval.
- Keep it concise (max 2 sentences).
- Return ONLY the rewritten search query, no explanation.

Rewritten search query:"""


class QueryRewriter:
    """
    Rewrites student queries to improve retrieval recall.

    Example:
        Input:  "deadlock kya hota hai"
        Output: "Explain deadlock in Operating Systems: necessary conditions,
                 prevention, avoidance using Banker's Algorithm"
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_MODEL,
    ) -> None:
        self.model = model
        self._client = None

        key = api_key or GROQ_API_KEY
        if key:
            try:
                from groq import Groq
                self._client = Groq(api_key=key)
                logger.info("QueryRewriter initialised (model=%s)", model)
            except ImportError:
                logger.warning("groq package not installed — pip install groq")
            except Exception as err:
                logger.warning("Groq client init failed: %s", err)
        else:
            logger.warning("GROQ_API_KEY not set — query rewriting disabled")

    def rewrite(self, query: str, context: Optional[str] = None) -> str:
        """
        Rewrite the query for better retrieval.

        Args:
            query: Original student query.
            context: Optional recent conversation context for pronoun resolution.

        Returns:
            Rewritten query string. Falls back to original if LLM unavailable.
        """
        if not query or not query.strip():
            return query

        if self._client is None:
            logger.debug("Query rewriter unavailable — returning original query")
            return query

        try:
            if context and context.strip():
                prompt = _REWRITE_PROMPT_WITH_CONTEXT.format(
                    context=context.strip(),
                    query=query.strip(),
                )
            else:
                prompt = _REWRITE_PROMPT.format(query=query.strip())

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            rewritten = response.choices[0].message.content.strip()
            logger.info("Query rewritten: %r → %r", query[:60], rewritten[:80])
            return rewritten or query
        except Exception as err:
            logger.warning("Query rewrite failed: %s — using original", err)
            return query
