"""Structured logging setup using loguru."""
from loguru import logger
import sys

def setup_logger():
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {name}:{function}:{line} | {message}", level="INFO")
    return logger
