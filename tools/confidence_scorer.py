"""Confidence scorer: converts reranker scores to confidence levels."""

from typing import List, Tuple

from config.settings import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD
from models.documents import ConfidenceLevel, Document
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceScorer:
    """
    Maps reranker scores to a human-readable confidence level.

    Thresholds (from config):
        > 0.85  → HIGH   ✅
        0.60–0.85 → MEDIUM ⚠️
        < 0.60  → LOW   ❌
    """

    def __init__(
        self,
        high_threshold: float = CONFIDENCE_HIGH_THRESHOLD,
        medium_threshold: float = CONFIDENCE_MEDIUM_THRESHOLD,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def score(
        self,
        ranked_docs: List[Tuple[Document, float]],
    ) -> Tuple[ConfidenceLevel, float]:
        """
        Compute overall confidence from reranker scores.

        Strategy: weighted average of top-3 scores
            (top doc contributes 50%, 2nd 30%, 3rd 20%)

        Args:
            ranked_docs: List of (Document, reranker_score) from CohereReranker.

        Returns:
            Tuple of (ConfidenceLevel enum, raw_score float 0-1).
        """
        if not ranked_docs:
            return ConfidenceLevel.LOW, 0.0

        weights = [0.5, 0.3, 0.2]
        top_docs = ranked_docs[:3]

        weighted_score = sum(
            score * weights[i]
            for i, (_, score) in enumerate(top_docs)
        )

        # Normalize if fewer than 3 docs
        used_weight = sum(weights[:len(top_docs)])
        if used_weight > 0:
            weighted_score /= used_weight

        level = self._to_level(weighted_score)
        logger.debug(
            "Confidence: raw_score=%.3f → %s (top scores: %s)",
            weighted_score,
            level,
            [f"{s:.3f}" for _, s in top_docs],
        )
        return level, round(weighted_score, 4)

    def _to_level(self, score: float) -> ConfidenceLevel:
        if score >= self.high_threshold:
            return ConfidenceLevel.HIGH
        if score >= self.medium_threshold:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def warning_message(self, level: ConfidenceLevel) -> str | None:
        """Return a user-facing warning for low/medium confidence answers."""
        if level == ConfidenceLevel.LOW:
            return (
                "⚠️ Low confidence — yeh answer knowledge base mein clearly nahi mila. "
                "Please textbook se verify karein."
            )
        if level == ConfidenceLevel.MEDIUM:
            return "ℹ️ Medium confidence — answer partially supported hai. Citation check karein."
        return None
