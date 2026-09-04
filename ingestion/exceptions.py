"""Exception hierarchy for document ingestion and parsing."""

from typing import Optional


class IngestionError(Exception):
    """Base exception for document ingestion and parsing errors."""

    def __init__(self, message: str, file_path: Optional[str] = None) -> None:
        self.file_path = file_path
        self.message = message
        full_msg = f"[{file_path}] {message}" if file_path else message
        super().__init__(full_msg)


class UnsupportedFileFormatError(IngestionError):
    """Raised when an unsupported file format is provided."""
    pass


class CorruptedFileError(IngestionError):
    """Raised when a file cannot be parsed by any extraction engine."""
    pass


class DocumentParsingError(IngestionError):
    """Raised when an error occurs during document extraction."""
    pass


class InvalidMetadataError(IngestionError):
    """Raised when document metadata validation fails."""
    pass
