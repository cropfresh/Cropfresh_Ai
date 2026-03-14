"""
Browser RAG Package — Re-exports for backward compatibility.
"""

from ai.rag.browser_rag_pkg.models import (
    Citation,
    CitedAnswer,
    ScrapeIntent,
    TargetSource,
)
from ai.rag.browser_rag_pkg.sources import AgriSourceSelector
from ai.rag.browser_rag_pkg.extractor import ContentExtractor, QualityFilter
from ai.rag.browser_rag_pkg.integration import BrowserRAGIntegration

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
