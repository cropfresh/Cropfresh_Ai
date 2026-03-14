"""
Browser RAG Integration — Backward compatibility redirect.

! This file is kept for backward compatibility. The actual implementation
! has been split into the `ai.rag.browser_rag_pkg` package.
! Import from `ai.rag.browser_rag_pkg` directly in new code.
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
