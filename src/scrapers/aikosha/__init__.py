"""
AI Kosha Client Package
=======================
Client for India's AI Kosha platform (indiaai.gov.in).
"""

from .models import AIKoshaCategory, AIKoshaDataset, AIKoshaSearchResult
from .client import AIKoshaClient
from .catalog import get_agri_catalog

__all__ = [
    "AIKoshaCategory",
    "AIKoshaDataset",
    "AIKoshaSearchResult",
    "AIKoshaClient",
    "get_agri_catalog",
]
