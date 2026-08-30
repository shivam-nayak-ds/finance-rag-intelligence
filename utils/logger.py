"""
SyllAIq — Centralized Logger
==============================
Single logger configuration for the entire project.
Import get_logger() in every module — never use print().
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import LOG_FORMAT, LOG_LEVEL, ROOT_DIR

LOGS_DIR = ROOT_DIR / "logs"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with both Console and Rotating File handlers.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Message")

    Args:
        name: Module name, typically __name__

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        formatter = logging.Formatter(LOG_FORMAT)

        # 1. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        logger.addHandler(console_handler)

        # 2. Rotating File Handler (logs/syllaiq.log)
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_file = LOGS_DIR / "syllaiq.log"
            file_handler = RotatingFileHandler(
                filename=str(log_file),
                maxBytes=10 * 1024 * 1024,  # 10 MB per log file
                backupCount=5,              # Keep up to 5 rotated backup logs
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as err:
            logger.warning(f"Could not initialize file logging: {err}")

    return logger
