"""
Multi-Query Expansion
"""

import time
from typing import Any

from loguru import logger

from .models import ExpandedQuery, QueryExpansionType, QueryProcessorConfig
from .prompts import MULTI_QUERY_PROMPT


async def multi_query_expand(llm: Any, config: QueryProcessorConfig, query: str) -> ExpandedQuery:
    """
    Generate multiple query variations.
    """
    start_time = time.time()
    expanded = await _multi_query_expand_internal(llm, config, query)

    return ExpandedQuery(
        original_query=query,
        expanded_queries=expanded,
        expansion_type=QueryExpansionType.MULTI_QUERY,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


async def _multi_query_expand_internal(llm: Any, config: QueryProcessorConfig, query: str) -> list[str]:
    """Generate multiple query variations."""
    if llm is None:
        return _rule_based_multi_query(config, query)

    try:
        count = config.multi_query_count
        prompt = MULTI_QUERY_PROMPT.format(query=query, count=count)
        response = await llm.agenerate([prompt])

        # Parse response into queries
        raw_queries = response.generations[0][0].text.strip().split('\n')
        queries = []

        for q in raw_queries:
            # Clean up numbering and whitespace
            q = q.strip()
            if q.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                q = q[2:].strip()
            if q and q != query:
                queries.append(q)

        return queries[:count]

    except Exception as e:
        logger.warning(f"Multi-query expansion failed: {e}")
        return _rule_based_multi_query(config, query)


def _rule_based_multi_query(config: QueryProcessorConfig, query: str) -> list[str]:
    """Simple rule-based query expansion."""
    query_lower = query.lower()
    expansions = []

    # Add synonym-based variations
    synonyms = {
        "tomato": ["tamatar", "tomatoes"],
        "onion": ["pyaj", "onions"],
        "potato": ["aloo", "potatoes"],
        "pest": ["insects", "bugs", "infestation"],
        "disease": ["infection", "blight", "rot"],
        "fertilizer": ["manure", "nutrients", "NPK"],
        "price": ["rate", "cost", "market value"],
        "yield": ["production", "harvest", "output"],
        "grow": ["cultivate", "plant", "farm"],
        "best": ["recommended", "ideal", "optimal"],
    }

    for word, syns in synonyms.items():
        if word in query_lower:
            for syn in syns[:1]:
                expansions.append(query_lower.replace(word, syn))

    # Add perspective variations
    if "how to" in query_lower:
        expansions.append(query_lower.replace("how to", "best practices for"))
        expansions.append(query_lower.replace("how to", "guide for"))

    if "what is" in query_lower:
        expansions.append(query_lower.replace("what is", "explain"))

    # Add location context
    if not any(loc in query_lower for loc in ["india", "karnataka", "maharashtra"]):
        expansions.append(query + " in India")

    return expansions[:config.multi_query_count]
