"""
Digital Twin Engine Package
===========================
Exports unified DigitalTwinEngine and factory functionally identical to module.
"""

from .core import DigitalTwinEngine, get_digital_twin_engine

__all__ = [
    "DigitalTwinEngine",
    "get_digital_twin_engine",
]
