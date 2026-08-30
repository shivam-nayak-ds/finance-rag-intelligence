"""
Academic Metadata Tagger (Production-Grade)
===========================================
Infers and tags syllabus unit numbers, canonical topic names, and exam metadata
with strict schema validation and confidence scoring.
"""

import re
from typing import Dict, Final, List, Optional, Protocol, Tuple, runtime_checkable

from config.settings import (
    RGPV_OS_UNIT_TOPICS,
    RGPV_OS_UNITS,
    TARGET_UNIVERSITY,
    V1_SUBJECT,
)
from ingestion.exceptions import InvalidMetadataError
from utils.logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class BaseMetadataTagger(Protocol):
    """Protocol defining the metadata tagging interface."""

    def tag_unit_and_topic(self, text: str) -> Tuple[int, str]:
        """Infers syllabus unit (1-5) and primary topic title."""
        ...

    def tag_pyq_metadata(
        self,
        question_text: str,
        year: Optional[int] = None,
        semester: Optional[int] = 4,
        marks: Optional[int] = 7,
        question_no: Optional[str] = None,
    ) -> Dict:
        """Constructs validated metadata dictionary for a PYQ item."""
        ...


class MetadataTagger:
    """
    Production-grade syllabus unit and topic classifier.
    Supports custom syllabus mappings via dependency injection.
    """

    _FALLBACK_RULES: Final[List[Tuple[int, List[str]]]] = [
        (4, ["deadlock", "semaphore", "mutual exclusion", "critical section", "dining philosopher", "ipc"]),
        (3, ["paging", "segmentation", "scheduling", "process", "thread", "virtual memory", "thrashing", "tcb"]),
        (2, ["disk scheduling", "file system", "directory structure", "contiguous allocation", "tape memory"]),
        (5, ["distributed", "network os", "unix", "linux", "windows", "dfs", "multiprocessor"]),
        (1, ["evolution", "operating system services", "system call", "batch processing", "spooling"]),
    ]

    def __init__(
        self,
        unit_topics: Optional[Dict[int, List[str]]] = None,
        unit_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Initializes tagger with optional custom topic mappings.
        """
        self._unit_topics = unit_topics or RGPV_OS_UNIT_TOPICS
        self._unit_names = unit_names or RGPV_OS_UNITS

    def tag_unit_and_topic(self, text: str) -> Tuple[int, str]:
        """
        Determines the most accurate syllabus unit and topic from text.

        Args:
            text: Input string (question statement or textbook section).

        Returns:
            Tuple of (unit_number, canonical_topic_name).
        """
        if not text or not isinstance(text, str):
            return 1, self._unit_names.get(1, "Introduction to Operating Systems")

        text_lower = text.lower()

        # Score units based on match count
        unit_scores: Dict[int, int] = {u: 0 for u in self._unit_topics.keys()}
        matched_topics: Dict[int, List[str]] = {u: [] for u in self._unit_topics.keys()}

        for unit_num, topics in self._unit_topics.items():
            for topic in topics:
                # Word-boundary check on short acronyms
                if len(topic) <= 4:
                    pattern = rf"\b{re.escape(topic)}\b"
                    match_count = len(re.findall(pattern, text_lower))
                else:
                    match_count = text_lower.count(topic)

                if match_count > 0:
                    unit_scores[unit_num] += match_count
                    matched_topics[unit_num].append(topic)

        # Pick unit with maximum match score
        best_unit = max(unit_scores, key=lambda u: unit_scores[u])

        # If zero matches, run fallback keyword rules
        if unit_scores[best_unit] == 0:
            for fallback_unit, keywords in self._FALLBACK_RULES:
                if any(kw in text_lower for kw in keywords):
                    best_unit = fallback_unit
                    break
            else:
                best_unit = 1

        # Select most specific matched topic title
        if matched_topics[best_unit]:
            primary_topic = max(matched_topics[best_unit], key=len).title()
        else:
            primary_topic = self._unit_names.get(best_unit, "Operating Systems").split(":")[0].strip()

        return best_unit, primary_topic

    def tag_pyq_metadata(
        self,
        question_text: str,
        year: Optional[int] = None,
        semester: Optional[int] = 4,
        marks: Optional[int] = 7,
        question_no: Optional[str] = None,
    ) -> Dict:
        """
        Builds validated metadata for an exam question with bounds checking.
        """
        if not question_text:
            raise InvalidMetadataError("Question text cannot be empty.")

        if year is not None and not (1990 <= year <= 2035):
            raise InvalidMetadataError(f"Invalid PYQ year: {year}", file_path=question_no)

        if marks is not None and not (1 <= marks <= 100):
            raise InvalidMetadataError(f"Invalid question marks: {marks}", file_path=question_no)

        unit, topic = self.tag_unit_and_topic(question_text)

        return {
            "source_type": "pyq",
            "subject": V1_SUBJECT,
            "university": TARGET_UNIVERSITY,
            "unit": unit,
            "topic": topic,
            "year": year,
            "semester": semester,
            "marks": marks,
            "question_no": question_no,
        }

    def tag_textbook_metadata(
        self,
        text: str,
        book_title: str = "Galvin OS 10th Ed",
        chapter: Optional[int] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
    ) -> Dict:
        """
        Builds validated metadata for a textbook section.
        """
        if not text:
            raise InvalidMetadataError("Textbook chunk text cannot be empty.")

        unit, topic = self.tag_unit_and_topic(text)
        return {
            "source_type": "textbook",
            "subject": V1_SUBJECT,
            "university": TARGET_UNIVERSITY,
            "book": book_title,
            "chapter": chapter,
            "page_start": page_start,
            "page_end": page_end or page_start,
            "unit": unit,
            "topic": topic,
        }
