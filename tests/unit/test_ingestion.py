"""Unit tests for document ingestion and text cleaning."""

import json
import pytest
from pathlib import Path

from ingestion.data_cleaner import DataCleaner
from ingestion.metadata_tagger import MetadataTagger
from ingestion.pdf_loader import PDFLoader
from ingestion.exceptions import (
    UnsupportedFileFormatError,
    CorruptedFileError,
    InvalidMetadataError,
)
from models.documents import Document, SourceType


class TestDataCleaner:
    """Tests for academic text cleaner."""

    def test_clean_text_removes_exam_watermarks(self):
        dirty = (
            "https://www.rgpvonline.com\n"
            "AD/CD/CS-405 (GS)\n"
            "Roll No.................\n"
            "Deadlock is a situation where processes wait for each other.\n"
            "P.T.O.\n"
            "[2]"
        )
        cleaned = DataCleaner.clean_text(dirty)
        assert "rgpvonline.com" not in cleaned
        assert "AD/CD/CS-405" not in cleaned
        assert "P.T.O." not in cleaned
        assert "Roll No" not in cleaned
        assert "[2]" not in cleaned
        assert "Deadlock is a situation" in cleaned

    def test_clean_text_normalizes_unicode(self):
        text_with_unicode = "Here\u2018s a \u201cquoted\u201d string \u2014 with dash."
        cleaned = DataCleaner.clean_text(text_with_unicode)
        assert cleaned == "Here's a \"quoted\" string -- with dash."

    def test_clean_pyq_question_removes_instructions(self):
        raw_pyq = (
            "Note : i) Attempt any five questions.\n"
            "What are the 4 necessary conditions for deadlock?"
        )
        cleaned = DataCleaner.clean_pyq_question(raw_pyq)
        assert "Attempt any" not in cleaned
        assert "What are the 4 necessary conditions for deadlock?" in cleaned

    def test_clean_text_handles_empty_input(self):
        assert DataCleaner.clean_text("") == ""
        assert DataCleaner.clean_text(None) == ""


class TestMetadataTagger:
    """Tests for metadata tagger."""

    def test_tag_deadlock_to_unit_4(self):
        tagger = MetadataTagger()
        text = "Explain Banker's Algorithm and 4 conditions of deadlock prevention."
        unit, topic = tagger.tag_unit_and_topic(text)
        assert unit == 4
        assert "Deadlock" in topic or "Banker" in topic

    def test_tag_paging_to_unit_3(self):
        tagger = MetadataTagger()
        text = "How does paging and segmentation translate logical address to physical address in virtual memory?"
        unit, topic = tagger.tag_unit_and_topic(text)
        assert unit == 3

    def test_tag_disk_scheduling_to_unit_2(self):
        tagger = MetadataTagger()
        text = "Compare SSTF, SCAN, and LOOK disk scheduling algorithms."
        unit, topic = tagger.tag_unit_and_topic(text)
        assert unit == 2

    def test_tag_pyq_metadata_validation(self):
        tagger = MetadataTagger()
        with pytest.raises(InvalidMetadataError):
            tagger.tag_pyq_metadata(question_text="", year=2023)

        with pytest.raises(InvalidMetadataError):
            tagger.tag_pyq_metadata(question_text="Valid text", year=1850)


class TestPDFLoader:
    """Tests for document loader."""

    def test_load_nonexistent_file_raises_error(self):
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_pdf("non_existent_file.pdf")

    def test_load_unsupported_format_raises_error(self, tmp_path: Path):
        bad_file = tmp_path / "test.docx"
        bad_file.write_text("dummy text")
        loader = PDFLoader()
        with pytest.raises(UnsupportedFileFormatError):
            loader.load_pdf(bad_file)

    def test_load_pyq_json(self, tmp_path: Path):
        sample_dataset = [
            {
                "paper_id": "TEST_2023",
                "year": 2023,
                "semester": 4,
                "questions": [
                    {
                        "question_no": "1.a",
                        "unit": 4,
                        "topic": "Deadlock",
                        "marks": 7,
                        "question_text": "What is Deadlock?",
                    }
                ],
            }
        ]
        json_file = tmp_path / "pyqs.json"
        json_file.write_text(json.dumps(sample_dataset), encoding="utf-8")

        loader = PDFLoader()
        docs = loader.load_pyq_json(json_file)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.PYQ
        assert docs[0].unit == 4
        assert docs[0].marks == 7
