"""
AI Kosha Client (Proxy)
=======================
This file exists for backward compatibility.
The actual implementation has been modularized into `src.scrapers.aikosha`.
"""

from src.scrapers.aikosha import (
    AIKoshaCategory,
    AIKoshaDataset,
    AIKoshaSearchResult,
    AIKoshaClient,
    get_agri_catalog,
)

__all__ = [
    "AIKoshaCategory",
    "AIKoshaDataset",
    "AIKoshaSearchResult",
    "AIKoshaClient",
    "get_agri_catalog",
]
