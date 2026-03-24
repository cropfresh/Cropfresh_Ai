"""
Digital Twin Engine Package
===========================
Exports unified DigitalTwinEngine and factory functionally identical to module.
"""

from .core import DigitalTwinEngine, get_digital_twin_engine
from .utils import (
    compute_transit_hours as _compute_transit_hours,
)
from .utils import (
    estimate_arrival_grade as _estimate_arrival_grade,
)
from .utils import (
    infer_arrival_defects as _infer_arrival_defects,
)

__all__ = [
    "DigitalTwinEngine",
    "get_digital_twin_engine",
    "_compute_transit_hours",
    "_estimate_arrival_grade",
    "_infer_arrival_defects",
]
