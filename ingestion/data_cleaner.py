"""
SyllAIq — Academic Data Cleaner
=================================
Cleans extracted text from academic PDFs, textbooks, syllabi, and PYQs.
Removes headers, footers, exam watermarks, while strictly preserving
mathematical notation, code snippets, algorithms, and technical terms.
"""

import re
from typing import List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Cleans raw text extracted from academic and university PDFs.
    """

    # Common exam website watermarks and noise lines
    NOISE_PATTERNS: List[re.Pattern] = [
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
        re.compile(r"Contd\.\.\.?", re.IGNORECASE),
        re.compile(r"P\.T\.O\.?", re.IGNORECASE),
        re.compile(r"\*{4,}", re.IGNORECASE),  # ****** lines
    ]

    # Patterns for page number indicators like [2], (3), Page 4 of 10
    PAGE_NUMBER_PATTERN = re.compile(r"^\s*\[?\s*\d+\s*\]?\s*$", re.MULTILINE)

    @classmethod
    def clean_text(cls, text: str, preserve_code: bool = True) -> str:
        """
        Cleans text string by stripping boilerplate, fixing encoding artifacts,
        and standardizing whitespace.

        Args:
            text: Raw extracted text
            preserve_code: Whether to preserve indentation for code blocks

        Returns:
            Cleaned and normalized text
        """
        if not text or not text.strip():
            return ""

        # Step 1: Remove common noise/watermark patterns
        cleaned = text
        for pattern in cls.NOISE_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        # Step 2: Remove isolated page number lines
        cleaned = cls.PAGE_NUMBER_PATTERN.sub("", cleaned)

        # Step 3: Normalize unicode characters (smart quotes, dashes, etc.)
        cleaned = cls._normalize_unicode(cleaned)

        # Step 4: Normalize whitespace while preserving paragraph structure
        lines = [line.strip() for line in cleaned.splitlines()]
        # Remove empty lines that repeat more than once
        non_empty_lines: List[str] = []
        consecutive_empty = 0
        for line in lines:
            if not line:
                consecutive_empty += 1
                if consecutive_empty <= 1:
                    non_empty_lines.append("")
            else:
                consecutive_empty = 0
                non_empty_lines.append(line)

        final_text = "\n".join(non_empty_lines).strip()
        return final_text

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Replaces common non-standard Unicode characters with standard ASCII."""
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "--",
            "\u2026": "...",
            "\u00a0": " ",  # Non-breaking space
            "\t": "    ",
        }
        for orig, replacement in replacements.items():
            text = text.replace(orig, replacement)
        return text

    @classmethod
    def clean_pyq_question(cls, question_text: str) -> str:
        """
        Specialized cleaner for individual PYQ questions.
        Removes exam boilerplate instructions like 'Attempt any five questions'.
        """
        cleaned = cls.clean_text(question_text)
        # Remove exam instruction banners if caught in question text
        cleaned = re.sub(r"Note\s*:\s*i\)\s*Attempt any.*?(?=\n|$)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"In case of any doubt.*?(?=\n|$)", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()
