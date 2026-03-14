"""
Query Rewriting
"""

import time
from loguru import logger
from typing import Any

from .models import ExpandedQuery, QueryExpansionType
from .prompts import REWRITE_PROMPT


async def rewrite_query(llm: Any, query: str) -> ExpandedQuery:
    """
    Query Rewriting.
    """
    start_time = time.time()
    rewritten = await _rewrite_internal(llm, query)
    
    return ExpandedQuery(
        original_query=query,
        rewritten_query=rewritten,
        expansion_type=QueryExpansionType.REWRITE,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


async def _rewrite_internal(llm: Any, query: str) -> str:
    """Rewrite query for better retrieval."""
    if llm is None:
        return _rule_based_rewrite(query)
    
    try:
        prompt = REWRITE_PROMPT.format(query=query)
        response = await llm.agenerate([prompt])
        return response.generations[0][0].text.strip()
        
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}")
        return _rule_based_rewrite(query)


def _rule_based_rewrite(query: str) -> str:
    """Simple rule-based query rewriting."""
    # Remove common filler phrases
    fillers = [
        "can you tell me",
        "i want to know",
        "please help me",
        "i need to understand",
        "what is the answer to",
        "could you explain",
    ]
    
    rewritten = query.lower()
    for filler in fillers:
        rewritten = rewritten.replace(filler, "").strip()
    
    # Standardize terminology
    replacements = {
        "tamatar": "tomato",
        "pyaj": "onion",
        "aloo": "potato",
        "gehu": "wheat",
        "dhan": "rice",
        "kapas": "cotton",
    }
    
    for hindi, english in replacements.items():
        rewritten = rewritten.replace(hindi, english)
    
    # Capitalize first letter
    if rewritten:
        rewritten = rewritten[0].upper() + rewritten[1:]
    
    return rewritten
