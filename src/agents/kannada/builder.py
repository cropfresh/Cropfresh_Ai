"""
Dynamic Kannada Context Builder
"""

from typing import Optional

from src.agents.kannada.administrative_terms import ADMIN_TERMS
from src.agents.kannada.agronomy_terms import AGRONOMY_TERMS
from src.agents.kannada.crop_varieties import CROP_VARIETIES_TERMS
from src.agents.kannada.equipment_terms import EQUIPMENT_TERMS
from src.agents.kannada.financial_terms import FINANCIAL_TERMS
from src.agents.kannada.guidelines import KANNADA_GUIDELINES
from src.agents.kannada.market_terms import MARKET_TERMS
from src.agents.kannada.platform_terms import PLATFORM_TERMS
from src.agents.kannada.soil_terms import SOIL_TERMS
from src.agents.kannada.weather_terms import WEATHER_TERMS


def get_kannada_context(domain_name: Optional[str] = None) -> str:
    """
    Assemble the Kannada context dynamically based on the agent's domain.

    Args:
        domain_name (str): The name/type of the agent requesting context
                           (e.g., 'agronomy', 'commerce', 'platform', 'general')

    Returns:
        str: The combined Kannada guidelines and relevant vocabulary terms.
    """
    parts = [KANNADA_GUIDELINES]

    domain = domain_name.lower() if domain_name else "general"

    if "agronomy" in domain:
        parts.append(AGRONOMY_TERMS)
        parts.append(CROP_VARIETIES_TERMS)
        parts.append(EQUIPMENT_TERMS)
        parts.append(WEATHER_TERMS)
        parts.append(SOIL_TERMS)
    elif "commerce" in domain or "price" in domain or "market" in domain:
        parts.append(MARKET_TERMS)
        parts.append(FINANCIAL_TERMS)
        parts.append(ADMIN_TERMS)
    elif "platform" in domain:
        parts.append(PLATFORM_TERMS)
        parts.append(FINANCIAL_TERMS)
        parts.append(ADMIN_TERMS)
    else:
        # General agent gets a mix of the most common platform and market terms
        parts.append(PLATFORM_TERMS)
        parts.append(MARKET_TERMS)
        parts.append(AGRONOMY_TERMS)

    return "\n\n".join(parts)
