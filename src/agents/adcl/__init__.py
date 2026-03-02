"""
ADCL Agent Package
==================
Adaptive Demand Crop List — weekly market intelligence for farmers.

Exports:
    ADCLAgent     : Main orchestrator (generate_weekly_report)
    get_adcl_agent: Factory function
    ADCLCrop      : Per-commodity data model
    WeeklyReport  : Full weekly report model
"""

from src.agents.adcl.engine import ADCLAgent, get_adcl_agent
from src.agents.adcl.models import ADCLCrop, WeeklyReport

__all__ = [
    "ADCLAgent",
    "get_adcl_agent",
    "ADCLCrop",
    "WeeklyReport",
]
