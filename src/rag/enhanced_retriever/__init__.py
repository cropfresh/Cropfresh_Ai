"""
Enhanced Retrieval Pipeline
=============================
Advanced retrieval strategies for improved context retrieval.

Implements:
- Parent Document Retriever: Store small chunks, retrieve full docs
- Sentence Window Retrieval: Expand context around matching sentences
- Auto-Merging Retrieval: Combine adjacent chunks
- MMR (Maximum Marginal Relevance): Diversity-focused retrieval

These techniques ensure relevant, complete, and diverse context
for answer generation.

Author: CropFresh AI Team
Version: 1.0.0
"""

from .models import (
    RetrievalStrategy,
    DocumentNode,
    RetrievalResult,
    EnhancedRetrieverConfig,
)
from .retriever import EnhancedRetriever, create_enhanced_retriever
from .parent_retriever import ParentDocumentRetriever
from .sentence_retriever import SentenceWindowRetriever
from .mmr_retriever import MMRRetriever

__all__ = [
    "RetrievalStrategy",
    "DocumentNode",
    "RetrievalResult",
    "EnhancedRetrieverConfig",
    "EnhancedRetriever",
    "create_enhanced_retriever",
    "ParentDocumentRetriever",
    "SentenceWindowRetriever",
    "MMRRetriever",
]
