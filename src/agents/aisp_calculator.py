"""
AISP Calculator
===============
All-Inclusive Sourcing Price (AISP) calculation engine.

Responsible for computing the total cost a buyer pays on CropFresh,
factoring in farmer payout, logistics, deadhead, handling, platform fees,
and a risk buffer — capped at Mandi modal price * 1.05.

Extracted from pricing_agent.py to keep both files under the 500-line rule.

Author: CropFresh AI Team
Version: 1.0.0
"""

from typing import Optional

from pydantic import BaseModel

# ── Domain model ─────────────────────────────────────────────────────────────

class AISPCalculation(BaseModel):
    """All-Inclusive Sourcing Price breakdown."""

    farmer_price_per_kg: float
    quantity_kg: float
    farmer_payout: float

    logistics_cost: float
    deadhead_surcharge: float
    handling_cost: float
    platform_fee: float
    platform_fee_pct: float
    risk_buffer: float
    risk_buffer_pct: float

    total_aisp: float
    aisp_per_kg: float
    mandi_cap_applied: bool


# ── Rate tables ───────────────────────────────────────────────────────────────

_DEADHEAD_FACTOR_TABLE = [
    (80, 101, 0.00),
    (60, 80,  0.10),
    (40, 60,  0.20),
    ( 0, 40,  0.35),
]

_LOGISTICS_RATE_TABLE = [
    (  0,  15, 1.5),
    ( 15,  50, 1.2),
    ( 50, 100, 1.0),
    (100, 1_000_000, 0.8),
]

_PLATFORM_FEE_TIERS = [
    (1000, float("inf"), 0.05),
    ( 500,        1000, 0.06),
    ( 100,         500, 0.07),
    (   0,         100, 0.08),
]

RISK_BUFFER_PCT        = 0.02
MANDI_CAP_MULTIPLIER   = 1.05
COLD_CHAIN_PREMIUM_PER_KM = 0.5  # ₹/km surcharge for refrigerated vehicles


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_logistics_rate(distance_km: float) -> float:
    for min_d, max_d, rate in _LOGISTICS_RATE_TABLE:
        if min_d <= distance_km < max_d:
            return rate
    return _LOGISTICS_RATE_TABLE[-1][2]


def _get_deadhead_factor(route_utilization_pct: float) -> float:
    for min_u, max_u, factor in _DEADHEAD_FACTOR_TABLE:
        if min_u <= route_utilization_pct < max_u:
            return factor
    return _DEADHEAD_FACTOR_TABLE[-1][2]


def _get_platform_fee_pct(quantity_kg: float) -> float:
    for min_q, max_q, fee in _PLATFORM_FEE_TIERS:
        if min_q <= quantity_kg < max_q:
            return fee
    return _PLATFORM_FEE_TIERS[-1][2]


# ── Public API ────────────────────────────────────────────────────────────────

def calculate_aisp(
    farmer_price_per_kg: float,
    quantity_kg: float,
    distance_km: float = 30,
    handling_per_kg: float = 0.5,
    mandi_modal_per_kg: Optional[float] = None,
    route_utilization_pct: float = 50.0,
    cold_chain: bool = False,
) -> AISPCalculation:
    """
    Calculate AISP = Farmer_Payout + Logistics + Deadhead + Handling +
                     Platform_Fee + Risk_Buffer, capped at Mandi * 1.05.

    Args:
        farmer_price_per_kg:    Price offered to the farmer (₹/kg).
        quantity_kg:            Cargo weight.
        distance_km:            One-way distance to delivery point.
        handling_per_kg:        Labour + packaging cost per kg.
        mandi_modal_per_kg:     Current mandi modal — activates price cap.
        route_utilization_pct:  % of truck capacity used (affects deadhead).
        cold_chain:             True if refrigerated vehicle is required.

    Returns:
        AISPCalculation with full cost breakdown.
    """
    if quantity_kg <= 0:
        raise ValueError("quantity_kg must be > 0")
    if farmer_price_per_kg < 0:
        raise ValueError("farmer_price_per_kg cannot be negative")
    if distance_km < 0:
        raise ValueError("distance_km cannot be negative")
    if not 0 <= route_utilization_pct <= 100:
        raise ValueError("route_utilization_pct must be 0–100")

    farmer_payout = farmer_price_per_kg * quantity_kg

    logistics_rate = _get_logistics_rate(distance_km)
    logistics_cost = distance_km * logistics_rate * quantity_kg / 1000
    if cold_chain:
        logistics_cost += distance_km * COLD_CHAIN_PREMIUM_PER_KM

    deadhead_factor   = _get_deadhead_factor(route_utilization_pct)
    deadhead_surcharge = logistics_cost * deadhead_factor

    handling_cost = handling_per_kg * quantity_kg

    subtotal         = farmer_payout + logistics_cost + deadhead_surcharge + handling_cost
    platform_fee_pct = _get_platform_fee_pct(quantity_kg)
    platform_fee     = subtotal * platform_fee_pct
    risk_buffer      = subtotal * RISK_BUFFER_PCT

    total_aisp  = subtotal + platform_fee + risk_buffer
    aisp_per_kg = total_aisp / quantity_kg

    mandi_cap_applied = False
    if mandi_modal_per_kg:
        cap = mandi_modal_per_kg * MANDI_CAP_MULTIPLIER
        if aisp_per_kg > cap:
            aisp_per_kg       = cap
            total_aisp        = cap * quantity_kg
            mandi_cap_applied = True

    return AISPCalculation(
        farmer_price_per_kg=farmer_price_per_kg,
        quantity_kg=quantity_kg,
        farmer_payout=farmer_payout,
        logistics_cost=logistics_cost,
        deadhead_surcharge=deadhead_surcharge,
        handling_cost=handling_cost,
        platform_fee=platform_fee,
        platform_fee_pct=platform_fee_pct,
        risk_buffer=risk_buffer,
        risk_buffer_pct=RISK_BUFFER_PCT,
        total_aisp=round(total_aisp, 2),
        aisp_per_kg=round(aisp_per_kg, 2),
        mandi_cap_applied=mandi_cap_applied,
    )
