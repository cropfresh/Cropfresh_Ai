"""
Logistics Router — Vehicle Selection
====================================
Vehicle type selection based on total weight and cold chain requirement.
Matches CropFresh cost model for <₹2.5/kg target.
"""

# * VEHICLE SELECTION MODULE
# NOTE: Vehicle costs from task11 research (base rate + per-km).
# NOTE: Cold chain required for perishables (e.g., leafy greens).

from __future__ import annotations

from dataclasses import dataclass

# * ═══════════════════════════════════════════════════════════════
# * Vehicle Configuration
# * ═══════════════════════════════════════════════════════════════

@dataclass
class VehicleConfig:
    """Vehicle type configuration for cost calculation."""

    vehicle_type: str
    capacity_kg: float
    base_rate_inr: float
    per_km_rate_inr: float
    cold_chain: bool


# * Vehicle types from task11 research
VEHICLES: list[VehicleConfig] = [
    VehicleConfig("2w_ev", 50.0, 50.0, 2.0, False),
    VehicleConfig("3w_auto", 300.0, 150.0, 4.0, False),
    VehicleConfig("tempo", 1500.0, 500.0, 8.0, False),
    VehicleConfig("cold_chain", 3000.0, 1500.0, 15.0, True),
]


def select_vehicle(
    total_weight_kg: float,
    cold_chain_required: bool = False,
) -> VehicleConfig:
    """
    Select optimal vehicle type by weight and cold chain need.

    Args:
        total_weight_kg: Total pickup weight in kg.
        cold_chain_required: True when commodity needs refrigeration.

    Returns:
        VehicleConfig for the selected vehicle type.
    """
    if cold_chain_required:
        candidates = [v for v in VEHICLES if v.cold_chain]
        if not candidates:
            return VEHICLES[-1]
    else:
        candidates = [v for v in VEHICLES if not v.cold_chain]

    for v in sorted(candidates, key=lambda x: x.capacity_kg):
        if total_weight_kg <= v.capacity_kg:
            return v

    return candidates[-1]
