"""LLM chain: calls Groq/Gemini and returns a structured RAGResult."""

import os
import time
from typing import List, Optional, Tuple

from config.settings import (
    GROQ_MODEL,
    GEMINI_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    PRIMARY_LLM,
)
from generation.prompt_templates import SYSTEM_PROMPT, build_context_block, build_prompt
from models.documents import Citation, ConfidenceLevel, Document, Intent
from models.responses import RAGResult
from utils.logger import get_logger

logger = get_logger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class LLMChain:
    """
    Calls the LLM with retrieved context and returns a RAGResult.

    Priority: Groq (LLaMA 3.3 70B) → Gemini 1.5 Flash → fallback error
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_model: str = GROQ_MODEL,
        gemini_model: str = GEMINI_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> None:
        self.groq_model = groq_model
        self.gemini_model = gemini_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._groq_client = None
        self._gemini_model_obj = None

        # Init Groq
        key = groq_api_key or GROQ_API_KEY
        if key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=key)
                logger.info("Groq client initialised (model=%s)", groq_model)
            except Exception as err:
                logger.warning("Groq init failed: %s", err)

        # Init Gemini fallback
        gkey = gemini_api_key or GEMINI_API_KEY
        if gkey:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gkey)
                self._gemini_model_obj = genai.GenerativeModel(gemini_model)
                logger.info("Gemini client initialised (model=%s)", gemini_model)
            except Exception as err:
                logger.warning("Gemini init failed: %s", err)

    # ──────────────────────────────────────────
    # Main generate method
    # ──────────────────────────────────────────

    def generate(
        self,
        query: str,
        documents: List[Document],
        ranked_docs: List[Tuple[Document, float]],
        intent: str = Intent.UNKNOWN,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        confidence_score: float = 0.0,
        warning: Optional[str] = None,
        retrieval_time_ms: float = 0.0,
        reranking_time_ms: float = 0.0,
        history_messages: Optional[List[dict]] = None,  # ← short-term memory
        personalization_hint: Optional[str] = None,     # ← long-term memory
    ) -> RAGResult:
        """
        Generate a cited answer using retrieved documents.

        Args:
            query: Original student query.
            documents: Top reranked documents to use as context.
            ranked_docs: (doc, score) pairs for citation generation.
            intent: Classified query intent.
            confidence: Pre-computed confidence level.
            confidence_score: Raw confidence score 0-1.
            warning: Optional warning string for low confidence.
            retrieval_time_ms: Time spent in retrieval phase.
            reranking_time_ms: Time spent in reranking phase.

        Returns:
            RAGResult with answer, citations, confidence, and timing info.
        """
        if not documents:
            return RAGResult(
                answer="Maafi karo, mujhe is question ka jawab knowledge base mein nahi mila. "
                       "Please apna textbook check karein.",
                citations=[],
                confidence=ConfidenceLevel.LOW,
                intent=intent,
                retrieval_failed=True,
                retrieval_time_ms=retrieval_time_ms,
                status="partial",
            )

        # Build context and prompt
        context = build_context_block(documents)
        user_prompt = build_prompt(query, context, intent)

        # Augment system prompt with personalization hint
        system = SYSTEM_PROMPT
        if personalization_hint:
            system = f"{SYSTEM_PROMPT}\n\n[Student Context]: {personalization_hint}"

        # Call LLM
        t0 = time.perf_counter()
        answer, total_tokens = self._call_llm(
            user_prompt,
            system=system,
            history=history_messages or [],
        )
        generation_ms = (time.perf_counter() - t0) * 1000

        # Build citations
        citations = [
            Citation.from_document(doc, i + 1)
            for i, doc in enumerate(documents)
        ]

        total_ms = retrieval_time_ms + reranking_time_ms + generation_ms

        return RAGResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            grounding_score=confidence_score,
            intent=intent,
            warning=warning,
            retrieval_time_ms=retrieval_time_ms,
            reranking_time_ms=reranking_time_ms,
            generation_time_ms=generation_ms,
            total_latency_ms=total_ms,
            total_tokens=total_tokens,
            status="success",
        )

    # ──────────────────────────────────────────
    # LLM call helpers
    # ──────────────────────────────────────────

    def _call_llm(
        self,
        user_prompt: str,
        system: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> Tuple[str, int]:
        """Try Groq first, fall back to Gemini."""
        if self._groq_client:
            try:
                return self._call_groq(user_prompt, system=system, history=history)
            except Exception as err:
                logger.warning("Groq call failed: %s — trying Gemini fallback", err)

        if self._gemini_model_obj:
            try:
                return self._call_gemini(user_prompt, system=system, history=history)
            except Exception as err:
                logger.error("Gemini fallback also failed: %s", err)

        return (
            "Abhi LLM service available nahi hai. Please thodi der baad try karein.",
            0,
        )

    def _call_groq(
        self,
        user_prompt: str,
        system: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> Tuple[str, int]:
        messages = [{"role": "system", "content": system or SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        response = self._groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        logger.info("Groq generation: %d tokens", tokens)
        return answer, tokens

    def _call_gemini(
        self,
        user_prompt: str,
        system: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> Tuple[str, int]:
        prompt_parts = [system or SYSTEM_PROMPT]
        if history:
            for msg in history:
                prompt_parts.append(f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}")
        prompt_parts.append(f"USER: {user_prompt}")
        full_prompt = "\n\n".join(prompt_parts)

        response = self._gemini_model_obj.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        answer = response.text.strip()
        logger.info("Gemini generation: answer length=%d chars", len(answer))
        return answer, len(answer.split())  # approx token count
