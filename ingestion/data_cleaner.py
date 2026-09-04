"""Text normalization and noise filtering utilities for academic documents."""

import re
from typing import Final, List, Optional, Protocol, Sequence, runtime_checkable

from utils.logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class BaseCleaner(Protocol):
    """Protocol defining the text cleaning interface."""

    def clean(self, text: str) -> str:
        """Cleans input text and returns sanitized output."""
        ...

    def clean_pyq(self, question_text: str) -> str:
        """Cleans PYQ-specific text."""
        ...


class DataCleaner:
    """Sanitizes text by removing watermarks, repeated headers/footers, and normalizing Unicode."""

    # Precompiled regex patterns for maximum execution speed
    _DEFAULT_NOISE_PATTERNS: Final[List[re.Pattern]] = [
        re.compile(r"https?://(?:www\.)?rgpvonline\.com[^\s]*", re.IGNORECASE),
        re.compile(r"www\.rgpvonline\.com", re.IGNORECASE),
        re.compile(r"AD/CD/CS(?:/SD)?-405\s*\(\w+\)", re.IGNORECASE),
        re.compile(r"CS-?405\s*\(\w+\)", re.IGNORECASE),
        re.compile(r"Total\s+No\.\s+of\s+Questions\s*:\s*\d+", re.IGNORECASE),
        re.compile(r"\[\s*Total\s+No\.\s+of\s+Printed\s+Pages\s*:\s*\d+\s*\]", re.IGNORECASE),
        re.compile(r"Roll\s+No[\.\s_:]*", re.IGNORECASE),
        re.compile(r"Grading\s+System\s*\(GS\)", re.IGNORECASE),
        re.compile(r"Maximum\s+Marks\s*:\s*\d+", re.IGNORECASE),
        re.compile(r"Time\s*:\s*Three\s+Hours", re.IGNORECASE),
        re.compile(r"\bContd\.\.\.?\b", re.IGNORECASE),
        re.compile(r"\bP\.T\.O\.?\b", re.IGNORECASE),
        re.compile(r"\*{4,}", re.IGNORECASE),
    ]

    _PAGE_NUM_PATTERN: Final[re.Pattern] = re.compile(r"^\s*\[?\s*\d+\s*\]?\s*$", re.MULTILINE)

    _EXAM_INSTRUCTION_PATTERNS: Final[List[re.Pattern]] = [
        re.compile(r"Note\s*:\s*i\)\s*Attempt any.*?(?=\n|$)", re.IGNORECASE),
        re.compile(r"In case of any doubt.*?(?=\n|$)", re.IGNORECASE),
        re.compile(r"All questions carry equal marks.*?(?=\n|$)", re.IGNORECASE),
    ]

    _UNICODE_MAP: Final[dict[str, str]] = {
        "\u2018": "'",   # Left single quote
        "\u2019": "'",   # Right single quote
        "\u201c": '"',   # Left double quote
        "\u201d": '"',   # Right double quote
        "\u2013": "-",   # En dash
        "\u2014": "--",  # Em dash
        "\u2026": "...", # Ellipsis
        "\u00a0": " ",   # Non-breaking space
        "\r\n": "\n",    # Normalize Windows CRLF
        "\r": "\n",
    }

    def __init__(self, custom_noise_patterns: Optional[Sequence[re.Pattern]] = None) -> None:
        """
        Initializes cleaner with default and optional custom noise patterns.
        """
        self._noise_patterns = list(self._DEFAULT_NOISE_PATTERNS)
        if custom_noise_patterns:
            self._noise_patterns.extend(custom_noise_patterns)

    def clean(self, text: str) -> str:
        """
        Cleans generic academic text.

        Args:
            text: Raw input string.

        Returns:
            Sanitized, paragraph-normalized string.
        """
        return self.clean_text(text)

    def clean_pyq(self, question_text: str) -> str:
        """
        Cleans PYQ exam question statements.

        Args:
            question_text: Raw question statement.

        Returns:
            Sanitized question statement.
        """
        return self.clean_pyq_question(question_text)

    @classmethod
    def clean_text(cls, text: Optional[str]) -> str:
        """
        Static classmethod for quick, stateless cleaning of generic academic text.

        Args:
            text: Raw input text.

        Returns:
            Normalized clean string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Unicode normalization
        cleaned = cls._normalize_unicode(text)

        # Step 2: Strip noise patterns
        for pattern in cls._DEFAULT_NOISE_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        # Step 3: Strip standalone page numbers
        cleaned = cls._PAGE_NUM_PATTERN.sub("", cleaned)

        # Step 4: Normalize whitespace preserving markdown/code paragraphs
        return cls._normalize_whitespace(cleaned)

    @classmethod
    def clean_pyq_question(cls, question_text: Optional[str]) -> str:
        """
        Strips exam metadata, banners, and noise from individual questions.

        Args:
            question_text: Raw question statement.

        Returns:
            Clean question statement.
        """
        if not question_text or not isinstance(question_text, str):
            return ""

        cleaned = cls.clean_text(question_text)
        if not cleaned:
            return ""

        for pattern in cls._EXAM_INSTRUCTION_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        return cleaned.strip()

    @classmethod
    def _normalize_unicode(cls, text: str) -> str:
        """Normalizes unicode typographical anomalies."""
        for src, target in cls._UNICODE_MAP.items():
            if src in text:
                text = text.replace(src, target)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        Trims trailing whitespace per line and compresses consecutive blank lines
        to a single empty line to preserve natural paragraph breaks.
        """
        lines = (line.strip() for line in text.splitlines())

        compact_lines: List[str] = []
        consecutive_blank = 0

        for line in lines:
            if not line:
                consecutive_blank += 1
                if consecutive_blank <= 1:
                    compact_lines.append("")
            else:
                consecutive_blank = 0
                compact_lines.append(line)

        return "\n".join(compact_lines).strip()
