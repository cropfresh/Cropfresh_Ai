"""
HyDE (Hypothetical Document Embeddings) Expansion
"""

import time
from loguru import logger
from typing import Any

from .models import ExpandedQuery, QueryExpansionType, QueryProcessorConfig
from .prompts import HYDE_PROMPT


async def hyde_expand(llm: Any, config: QueryProcessorConfig, query: str) -> ExpandedQuery:
    """
    HyDE (Hypothetical Document Embeddings).
    """
    start_time = time.time()
    hypothetical_doc = await _hyde_expand_internal(llm, config, query)
    
    return ExpandedQuery(
        original_query=query,
        hypothetical_doc=hypothetical_doc,
        expansion_type=QueryExpansionType.HYDE,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


async def _hyde_expand_internal(llm: Any, config: QueryProcessorConfig, query: str) -> str:
    """Generate hypothetical document for HyDE."""
    if llm is None:
        return _rule_based_hyde(query)
    
    try:
        prompt = HYDE_PROMPT.format(query=query)
        response = await llm.agenerate([prompt])
        hypothetical = response.generations[0][0].text.strip()
        
        # Truncate if too long
        if len(hypothetical) > config.hyde_max_length * 4:
            hypothetical = hypothetical[:config.hyde_max_length * 4]
        
        return hypothetical
        
    except Exception as e:
        logger.warning(f"HyDE expansion failed: {e}")
        return _rule_based_hyde(query)


def _rule_based_hyde(query: str) -> str:
    """Simple rule-based hypothetical document generation."""
    query_lower = query.lower()
    
    if "how" in query_lower or "what" in query_lower:
        return f"To address '{query}', farmers should consider several factors. " \
               f"Based on agricultural best practices in India, the recommended approach involves " \
               f"proper planning, selecting appropriate varieties, and following scientific methods. " \
               f"Key considerations include soil preparation, irrigation management, and pest control."
    elif "when" in query_lower:
        return f"The timing for '{query}' depends on the crop season and local conditions. " \
               f"In India, Kharif season (June-October) and Rabi season (October-March) have different cycles. " \
               f"Consult local KVK for region-specific guidance."
    elif "why" in query_lower:
        return f"Understanding '{query}' is crucial for successful farming. " \
               f"The reasons involve multiple factors including soil health, climate, and market conditions. " \
               f"Scientific research has established clear guidelines for farmers."
    else:
        return f"Regarding '{query}', agricultural experts recommend following established guidelines. " \
               f"Proper implementation involves understanding crop requirements, local conditions, and market dynamics. " \
               f"Farmers should consult agricultural extension services for personalized advice."
