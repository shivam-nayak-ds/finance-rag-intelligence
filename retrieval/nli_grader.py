"""NLI-based relevance grader using cross-encoder for Self-RAG groundedness checks."""

from typing import List, Optional, Tuple

from config.settings import NLI_GROUNDEDNESS_THRESHOLD, NLI_RELEVANCE_THRESHOLD
from models.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


class NLIGrader:
    """
    Uses a cross-encoder NLI model to grade:
      1. Document Relevance — is this chunk relevant to the query?
      2. Answer Groundedness — is the answer supported by the retrieved chunks?

    Falls back gracefully if model unavailable.
    """

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        relevance_threshold: float = NLI_RELEVANCE_THRESHOLD,
        groundedness_threshold: float = NLI_GROUNDEDNESS_THRESHOLD,
    ) -> None:
        self.relevance_threshold = relevance_threshold
        self.groundedness_threshold = groundedness_threshold
        self._pipeline = None

        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=-1,  # CPU
            )
            logger.info("NLI grader loaded: %s", model_name)
        except Exception as err:
            logger.warning("NLI grader unavailable (%s) — grading disabled", err)

    # ──────────────────────────────────────────
    # 1. Document Relevance Grading
    # ──────────────────────────────────────────

    def grade_documents(
        self,
        query: str,
        documents: List[Document],
        threshold: Optional[float] = None,
    ) -> Tuple[List[Document], List[Document]]:
        """
        Filter documents into relevant and irrelevant buckets.

        Args:
            query: User query.
            documents: Candidate document chunks.
            threshold: Override default relevance threshold.

        Returns:
            Tuple of (relevant_docs, irrelevant_docs).
        """
        if not self._pipeline or not documents:
            logger.debug("NLI grader unavailable — returning all docs as relevant")
            return documents, []

        cutoff = threshold if threshold is not None else self.relevance_threshold
        relevant, irrelevant = [], []

        for doc in documents:
            score = self._nli_score(premise=doc.text[:512], hypothesis=query)
            doc_copy = doc.model_copy()
            doc_copy.nli_score = score

            if score >= cutoff:
                relevant.append(doc_copy)
            else:
                irrelevant.append(doc_copy)

        logger.info(
            "NLI grading: %d total → %d relevant, %d irrelevant (threshold=%.2f)",
            len(documents), len(relevant), len(irrelevant), cutoff,
        )
        return relevant, irrelevant

    # ──────────────────────────────────────────
    # 2. Answer Groundedness Check
    # ──────────────────────────────────────────

    def check_groundedness(
        self,
        answer: str,
        documents: List[Document],
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Check whether an answer is grounded in the retrieved documents.

        Args:
            answer: Generated answer text.
            documents: Supporting document chunks.
            threshold: Override default groundedness threshold.

        Returns:
            Tuple of (is_grounded: bool, score: float).
        """
        # 1. Fail-secure if model is missing or documents list is empty
        if not self._pipeline or not documents:
            logger.debug("NLI grader unavailable or empty documents — failing secure (ungrounded)")
            return False, 0.0

        cutoff = threshold if threshold is not None else self.groundedness_threshold

        # 2. Use full text chunks instead of character cutoffs
        combined_context = " ".join(d.text for d in documents[:3])
        score = self._nli_score(premise=combined_context, hypothesis=answer)

        is_grounded = score >= cutoff
        logger.info(
            "Groundedness check: score=%.3f threshold=%.2f → %s",
            score, cutoff, "GROUNDED" if is_grounded else "HALLUCINATION",
        )
        return is_grounded, score

    # ──────────────────────────────────────────
    # Internal NLI scoring
    # ──────────────────────────────────────────

    def _nli_score(self, premise: str, hypothesis: str) -> float:
        """
        Run NLI model and return entailment probability.
        Labels vary by model — handles both 'entailment' and 'ENTAILMENT'.
        """
        try:
            result = self._pipeline(
                f"{premise} [SEP] {hypothesis}",
                truncation=True,
                max_length=512,
            )
            # result is list of {label, score}
            for item in result:
                if "entail" in item["label"].lower():
                    return float(item["score"])
            # If no entailment label found, return lowest score
            return 0.0
        except Exception as err:
            logger.warning("NLI score failed: %s", err)
            return 0.0  # Fail secure on exception
