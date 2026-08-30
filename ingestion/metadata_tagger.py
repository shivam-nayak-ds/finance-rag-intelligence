"""
SyllAIq — Metadata Tagger
==========================
Automatically tags extracted documents and chunks with:
- Academic Subject ("Operating Systems")
- RGPV Syllabus Unit (1 to 5)
- Canonical Topic (e.g. "Deadlock", "Banker's Algorithm", "CPU Scheduling")
- Document Type (Textbook, PYQ, Syllabus)
"""

import re
from typing import Optional, Tuple, List, Dict
from config.settings import (
    V1_SUBJECT,
    TARGET_UNIVERSITY,
    RGPV_OS_UNITS,
    RGPV_OS_UNIT_TOPICS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class MetadataTagger:
    """
    Tags documents with authoritative syllabus units and topics.
    """

    @classmethod
    def tag_unit_and_topic(cls, text: str) -> Tuple[int, str]:
        """
        Determines the most likely RGPV OS Unit (1-5) and primary topic
        from text content using keyword density and keyword matching.

        Args:
            text: Chunk or question text

        Returns:
            Tuple of (unit_number, primary_topic)
        """
        if not text:
            return 1, "General"

        text_lower = text.lower()

        # Score each unit based on matches against its known topics
        unit_scores: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        matched_topics: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}

        for unit_num, topics in RGPV_OS_UNIT_TOPICS.items():
            for topic in topics:
                # Use word-boundary regex for short keywords to prevent substring false positives
                if len(topic) <= 4:
                    pattern = rf"\b{re.escape(topic)}\b"
                    matches = len(re.findall(pattern, text_lower))
                else:
                    matches = text_lower.count(topic)

                if matches > 0:
                    unit_scores[unit_num] += matches
                    matched_topics[unit_num].append(topic)

        # Find the unit with highest score
        best_unit = max(unit_scores, key=lambda u: unit_scores[u])
        if unit_scores[best_unit] == 0:
            # Fallback heuristic checks
            if any(w in text_lower for w in ["deadlock", "semaphore", "mutual exclusion", "dining philosopher"]):
                best_unit = 4
            elif any(w in text_lower for w in ["paging", "segmentation", "scheduling", "process", "thread", "virtual memory"]):
                best_unit = 3
            elif any(w in text_lower for w in ["disk", "file", "directory", "allocation"]):
                best_unit = 2
            elif any(w in text_lower for w in ["distributed", "unix", "linux", "windows", "network os"]):
                best_unit = 5
            else:
                best_unit = 1

        # Select most specific matched topic or fallback
        if matched_topics[best_unit]:
            # Pick longest matching topic phrase for highest specificity
            primary_topic = max(matched_topics[best_unit], key=len).title()
        else:
            primary_topic = RGPV_OS_UNITS.get(best_unit, "Operating Systems").split(":")[0].strip()

        return best_unit, primary_topic

    @classmethod
    def tag_pyq_metadata(
        cls,
        question_text: str,
        year: Optional[int] = None,
        semester: Optional[int] = 4,
        marks: Optional[int] = 7,
        question_no: Optional[str] = None,
    ) -> Dict:
        """
        Enriches a PYQ question with standard metadata.
        """
        unit, topic = cls.tag_unit_and_topic(question_text)
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

    @classmethod
    def tag_textbook_metadata(
        cls,
        text: str,
        book_title: str = "Galvin OS 10th Ed",
        chapter: Optional[int] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
    ) -> Dict:
        """
        Enriches a textbook chunk with standard metadata.
        """
        unit, topic = cls.tag_unit_and_topic(text)
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
