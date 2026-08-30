"""
SyllAIq — Centralized Logger
==============================
Single logger configuration for the entire project.
Import get_logger() in every module — never use print().
"""

import logging
import sys
from config.settings import LOG_LEVEL, LOG_FORMAT


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Pipeline started")

    Args:
        name: Module name, typically __name__

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
