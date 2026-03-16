"""
Browser RAG Package — Re-exports for backward compatibility.
"""

from src.rag.browser_rag_pkg.extractor import ContentExtractor, QualityFilter
from src.rag.browser_rag_pkg.integration import BrowserRAGIntegration
from src.rag.browser_rag_pkg.models import (
    Citation,
    CitedAnswer,
    ScrapeIntent,
    TargetSource,
)
from src.rag.browser_rag_pkg.sources import AgriSourceSelector

__all__ = [
    "Citation",
    "CitedAnswer",
    "ScrapeIntent",
    "TargetSource",
    "AgriSourceSelector",
    "ContentExtractor",
    "QualityFilter",
    "BrowserRAGIntegration",
]
