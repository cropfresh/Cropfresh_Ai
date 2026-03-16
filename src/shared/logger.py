"""Structured logging helpers for CropFresh services and scripts."""

from __future__ import annotations

import sys

from loguru import logger

_LOGGER_CONFIGURED = False


def setup_logger(level: str = "INFO"):
    """Configure the shared loguru logger once and return it."""
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        logger.remove()
        logger.add(
            sys.stderr,
            format="{time:HH:mm:ss} | {level:<7} | {name}:{function}:{line} | {message}",
            level=level,
        )
        _LOGGER_CONFIGURED = True
    return logger
