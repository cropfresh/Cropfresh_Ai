"""
Query Processor Package
=======================
Advanced query processing techniques for improved retrieval.
"""

from typing import Optional
from typing import Any

from .models import QueryExpansionType, ExpandedQuery, QueryProcessorConfig
from .processor import AdvancedQueryProcessor


def create_query_processor(
    llm: Any = None,
    config: Optional[QueryProcessorConfig] = None,
) -> AdvancedQueryProcessor:
    """
    Create an advanced query processor.
    """
    return AdvancedQueryProcessor(llm=llm, config=config)


__all__ = [
    "QueryExpansionType",
    "ExpandedQuery",
    "QueryProcessorConfig",
    "AdvancedQueryProcessor",
    "create_query_processor",
]
