"""
Digital Twin Engine Package
============================
AI-powered produce quality tracking for CropFresh dispute resolution.

Exports:
    DigitalTwinEngine   — main engine class
    get_digital_twin_engine — factory function
    DigitalTwin         — departure snapshot dataclass
    ArrivalData         — arrival state dataclass
    DiffReport          — diff analysis result dataclass
    LiabilityResult     — liability determination result
"""

from src.agents.digital_twin.engine import DigitalTwinEngine, get_digital_twin_engine
from src.agents.digital_twin.liability import LiabilityResult, determine_liability
from src.agents.digital_twin.models import ArrivalData, DiffReport, DigitalTwin

__all__ = [
    "DigitalTwinEngine",
    "get_digital_twin_engine",
    "DigitalTwin",
    "ArrivalData",
    "DiffReport",
    "LiabilityResult",
    "determine_liability",
]
