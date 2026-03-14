"""
Step-Back Query Expansion
"""

import time
from typing import Any

from loguru import logger

from .models import ExpandedQuery, QueryExpansionType
from .prompts import STEP_BACK_PROMPT


async def step_back_expand(llm: Any, query: str) -> ExpandedQuery:
    """
    Step-Back Prompting.
    """
    start_time = time.time()
    step_back = await _step_back_internal(llm, query)

    return ExpandedQuery(
        original_query=query,
        step_back_query=step_back,
        expansion_type=QueryExpansionType.STEP_BACK,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


async def _step_back_internal(llm: Any, query: str) -> str:
    """Generate step-back question."""
    if llm is None:
        return _rule_based_step_back(query)

    try:
        prompt = STEP_BACK_PROMPT.format(query=query)
        response = await llm.agenerate([prompt])
        return response.generations[0][0].text.strip()

    except Exception as e:
        logger.warning(f"Step-back expansion failed: {e}")
        return _rule_based_step_back(query)


def _rule_based_step_back(query: str) -> str:
    """Simple rule-based step-back question."""
    query_lower = query.lower()

    # Extract crop if mentioned
    crops = ["tomato", "onion", "potato", "rice", "wheat", "cotton", "maize"]
    mentioned_crop = None
    for crop in crops:
        if crop in query_lower:
            mentioned_crop = crop
            break

    # Generate abstract question based on topic
    if "pest" in query_lower or "insect" in query_lower:
        if mentioned_crop:
            return f"What are the general principles of integrated pest management for {mentioned_crop}?"
        return "What are the general principles of integrated pest management in vegetable crops?"

    elif "disease" in query_lower or "blight" in query_lower:
        return "What are the common causes and prevention methods for plant diseases?"

    elif "price" in query_lower or "market" in query_lower:
        return "What factors affect agricultural commodity prices in Indian markets?"

    elif "yield" in query_lower or "production" in query_lower:
        return "What are the key factors that determine crop yield and productivity?"

    elif "fertilizer" in query_lower or "nutrient" in query_lower:
        return "What are the nutrient requirements and fertilizer management principles for crops?"

    elif "water" in query_lower or "irrigat" in query_lower:
        return "What are the principles of efficient irrigation management in agriculture?"

    else:
        if mentioned_crop:
            return f"What are the complete cultivation practices for {mentioned_crop} in India?"
        return "What are the fundamental principles of successful agricultural practices in India?"
