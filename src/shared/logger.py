"""Structured logging setup using loguru."""
import sys

from loguru import logger


def setup_logger():
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {name}:{function}:{line} | {message}", level="INFO")
    return logger
