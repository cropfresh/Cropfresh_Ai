"""
AI Kosha Client Package
=======================
Client for India's AI Kosha platform (indiaai.gov.in).
"""

from .catalog import get_agri_catalog
from .client import AIKoshaClient
from .models import AIKoshaCategory, AIKoshaDataset, AIKoshaSearchResult

__all__ = [
    "AIKoshaCategory",
    "AIKoshaDataset",
    "AIKoshaSearchResult",
    "AIKoshaClient",
    "get_agri_catalog",
]
