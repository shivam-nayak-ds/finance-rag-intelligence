"""
SyllAIq Ingestion Exceptions
============================
Custom exception hierarchy for the data ingestion and document processing pipeline.
"""

from typing import Optional


class IngestionError(Exception):
    """Base exception for all document ingestion and parsing errors."""

    def __init__(self, message: str, file_path: Optional[str] = None) -> None:
        self.file_path = file_path
        self.message = message
        full_msg = f"[{file_path}] {message}" if file_path else message
        super().__init__(full_msg)


class UnsupportedFileFormatError(IngestionError):
    """Raised when an unsupported file extension or format is passed to a loader."""
    pass


class CorruptedFileError(IngestionError):
    """Raised when a PDF or data file is malformed or unreadable by all parsing engines."""
    pass


class DocumentParsingError(IngestionError):
    """Raised when a document or page fails during text/metadata extraction."""
    pass


class InvalidMetadataError(IngestionError):
    """Raised when document metadata fails validation constraints."""
    pass
