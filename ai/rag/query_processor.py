"""
Query Processor Proxy
=====================
Proxy to avoid duplicating code between ai/rag and src/rag.
Features are imported directly from src/rag/query_processor.
"""

from src.rag.query_processor import (
    QueryExpansionType,
    ExpandedQuery,
    QueryProcessorConfig,
    AdvancedQueryProcessor,
    create_query_processor,
)

__all__ = [
    "QueryExpansionType",
    "ExpandedQuery",
    "QueryProcessorConfig",
    "AdvancedQueryProcessor",
    "create_query_processor",
]
