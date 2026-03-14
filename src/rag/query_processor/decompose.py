"""
Query Decomposition
"""

import time
from loguru import logger
from typing import Any

from .models import ExpandedQuery, QueryExpansionType, QueryProcessorConfig
from .prompts import DECOMPOSE_PROMPT


async def decompose_query(llm: Any, config: QueryProcessorConfig, query: str) -> ExpandedQuery:
    """
    Query Decomposition.
    """
    start_time = time.time()
    sub_queries = await _decompose_internal(llm, config, query)
    
    return ExpandedQuery(
        original_query=query,
        sub_queries=sub_queries,
        expansion_type=QueryExpansionType.DECOMPOSE,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


async def _decompose_internal(llm: Any, config: QueryProcessorConfig, query: str) -> list[str]:
    """Decompose query into sub-queries."""
    if llm is None:
        return _rule_based_decompose(query)
    
    try:
        prompt = DECOMPOSE_PROMPT.format(query=query)
        response = await llm.agenerate([prompt])
        
        # Parse sub-queries
        raw = response.generations[0][0].text.strip().split('\n')
        sub_queries = []
        
        for q in raw:
            q = q.strip()
            if q.startswith(('1.', '2.', '3.', '-', '*')):
                q = q[2:].strip()
            if q and q != query:
                sub_queries.append(q)
        
        return sub_queries[:config.max_sub_queries]
        
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")
        return _rule_based_decompose(query)


def _rule_based_decompose(query: str) -> list[str]:
    """Simple rule-based query decomposition."""
    query_lower = query.lower()
    
    # Check for compound queries (and, also, as well as)
    if " and " in query_lower:
        parts = query_lower.split(" and ")
        return [p.strip().capitalize() + "?" for p in parts if len(p) > 10]
    
    # Check for multi-aspect queries
    if "how to" in query_lower and ("when" in query_lower or "where" in query_lower):
        return [
            query_lower.split("when")[0].strip() + "?",
            "When is the best time for " + query_lower.split("how to")[-1].split("when")[0].strip() + "?",
        ]
    
    # Default: return empty (query is simple)
    return []
