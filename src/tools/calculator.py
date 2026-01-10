"""
Calculator Tool
===============
Calculation utilities for agricultural commerce.

Provides:
- AISP (All-Inclusive Sourcing Price) calculations
- Yield and profit estimations
- Unit conversions (quintal <-> kg)
- Margin calculations

Author: CropFresh AI Team
Version: 2.0.0
"""

from typing import Optional

from loguru import logger
from pydantic import BaseModel

from src.tools.registry import get_tool_registry


class AISPResult(BaseModel):
    """AISP calculation result."""
    
    farmer_price_per_kg: float
    quantity_kg: float
    farmer_payout: float
    
    logistics_cost: float
    handling_cost: float
    platform_fee: float
    platform_fee_pct: float
    
    total_aisp: float
    aisp_per_kg: float
    
    # Margins
    buyer_total: float
    margin_over_farmer: float


class YieldEstimate(BaseModel):
    """Crop yield estimation."""
    
    crop: str
    area_acres: float
    expected_yield_kg: float
    yield_range_min: float
    yield_range_max: float
    
    # Revenue estimate
    price_per_kg: float
    estimated_revenue: float
    revenue_range_min: float
    revenue_range_max: float


class CalculatorTool:
    """
    Agricultural calculation utilities.
    
    Provides standardized calculations for:
    - AISP (All-Inclusive Sourcing Price)
    - Yield estimations
    - Profit margins
    - Unit conversions
    
    Usage:
        calc = CalculatorTool()
        aisp = calc.calculate_aisp(25, 500, 30)  # ₹25/kg, 500kg, 30km
    """
    
    # Platform fee tiers (quantity_kg, fee_pct)
    PLATFORM_FEE_TIERS = [
        (100, 0.08),    # 0-100 kg: 8%
        (500, 0.06),    # 100-500 kg: 6%
        (float("inf"), 0.04),  # 500+ kg: 4%
    ]
    
    # Logistics rates (distance_km, rate_per_kg)
    LOGISTICS_RATES = [
        (25, 2.0),     # 0-25 km: ₹2/kg
        (50, 2.5),     # 25-50 km: ₹2.5/kg
        (100, 3.0),    # 50-100 km: ₹3/kg
        (float("inf"), 4.0),  # 100+ km: ₹4/kg
    ]
    
    # Typical yields (kg per acre)
    CROP_YIELDS = {
        "tomato": {"min": 8000, "max": 20000, "typical": 15000},
        "potato": {"min": 6000, "max": 15000, "typical": 10000},
        "onion": {"min": 5000, "max": 12000, "typical": 8000},
        "carrot": {"min": 6000, "max": 12000, "typical": 9000},
        "cabbage": {"min": 10000, "max": 25000, "typical": 18000},
        "capsicum": {"min": 4000, "max": 10000, "typical": 7000},
        "beans": {"min": 3000, "max": 6000, "typical": 4000},
        "cauliflower": {"min": 8000, "max": 16000, "typical": 12000},
    }
    
    def calculate_aisp(
        self,
        farmer_price_per_kg: float,
        quantity_kg: float,
        distance_km: float = 30,
        handling_per_kg: float = 0.5,
    ) -> AISPResult:
        """
        Calculate All-Inclusive Sourcing Price.
        
        AISP = Farmer_Payout + Logistics + Handling + Platform_Fee
        
        Args:
            farmer_price_per_kg: Price paid to farmer (₹/kg)
            quantity_kg: Quantity in kg
            distance_km: Distance from farm to buyer
            handling_per_kg: Handling cost per kg
            
        Returns:
            AISPResult with full breakdown
        """
        # Farmer payout
        farmer_payout = farmer_price_per_kg * quantity_kg
        
        # Logistics cost
        logistics_rate = self._get_logistics_rate(distance_km)
        logistics_cost = logistics_rate * quantity_kg
        
        # Handling cost
        handling_cost = handling_per_kg * quantity_kg
        
        # Platform fee (on subtotal)
        subtotal = farmer_payout + logistics_cost + handling_cost
        platform_fee_pct = self._get_platform_fee(quantity_kg)
        platform_fee = subtotal * platform_fee_pct
        
        # Total AISP
        total_aisp = subtotal + platform_fee
        aisp_per_kg = total_aisp / quantity_kg if quantity_kg > 0 else 0
        
        # Margin calculation
        margin = aisp_per_kg - farmer_price_per_kg
        
        return AISPResult(
            farmer_price_per_kg=farmer_price_per_kg,
            quantity_kg=quantity_kg,
            farmer_payout=round(farmer_payout, 2),
            logistics_cost=round(logistics_cost, 2),
            handling_cost=round(handling_cost, 2),
            platform_fee=round(platform_fee, 2),
            platform_fee_pct=platform_fee_pct,
            total_aisp=round(total_aisp, 2),
            aisp_per_kg=round(aisp_per_kg, 2),
            buyer_total=round(total_aisp, 2),
            margin_over_farmer=round(margin, 2),
        )
    
    def estimate_yield(
        self,
        crop: str,
        area_acres: float,
        price_per_kg: float = 0,
    ) -> YieldEstimate:
        """
        Estimate crop yield for given area.
        
        Args:
            crop: Crop name
            area_acres: Farm area in acres
            price_per_kg: Current market price (for revenue estimate)
            
        Returns:
            YieldEstimate with production estimates
        """
        crop_lower = crop.lower()
        yields = self.CROP_YIELDS.get(
            crop_lower,
            {"min": 5000, "max": 15000, "typical": 10000}
        )
        
        expected = yields["typical"] * area_acres
        min_yield = yields["min"] * area_acres
        max_yield = yields["max"] * area_acres
        
        return YieldEstimate(
            crop=crop.title(),
            area_acres=area_acres,
            expected_yield_kg=round(expected, 0),
            yield_range_min=round(min_yield, 0),
            yield_range_max=round(max_yield, 0),
            price_per_kg=price_per_kg,
            estimated_revenue=round(expected * price_per_kg, 0) if price_per_kg else 0,
            revenue_range_min=round(min_yield * price_per_kg, 0) if price_per_kg else 0,
            revenue_range_max=round(max_yield * price_per_kg, 0) if price_per_kg else 0,
        )
    
    def _get_logistics_rate(self, distance_km: float) -> float:
        """Get logistics rate based on distance."""
        for max_dist, rate in self.LOGISTICS_RATES:
            if distance_km <= max_dist:
                return rate
        return self.LOGISTICS_RATES[-1][1]
    
    def _get_platform_fee(self, quantity_kg: float) -> float:
        """Get platform fee percentage based on quantity."""
        for max_qty, fee in self.PLATFORM_FEE_TIERS:
            if quantity_kg <= max_qty:
                return fee
        return self.PLATFORM_FEE_TIERS[-1][1]
    
    @staticmethod
    def kg_to_quintal(kg: float) -> float:
        """Convert kg to quintal (1 quintal = 100 kg)."""
        return kg / 100
    
    @staticmethod
    def quintal_to_kg(quintal: float) -> float:
        """Convert quintal to kg."""
        return quintal * 100
    
    @staticmethod
    def price_per_kg_to_quintal(price_per_kg: float) -> float:
        """Convert price per kg to per quintal."""
        return price_per_kg * 100
    
    @staticmethod
    def price_per_quintal_to_kg(price_per_quintal: float) -> float:
        """Convert price per quintal to per kg."""
        return price_per_quintal / 100


# Tool functions for registry
def _calculate_aisp(
    farmer_price_per_kg: float,
    quantity_kg: float,
    distance_km: float = 30,
) -> dict:
    """Calculate AISP for a transaction."""
    calc = CalculatorTool()
    result = calc.calculate_aisp(farmer_price_per_kg, quantity_kg, distance_km)
    return result.model_dump()


def _estimate_yield(
    crop: str,
    area_acres: float,
    price_per_kg: float = 0,
) -> dict:
    """Estimate crop yield for given area."""
    calc = CalculatorTool()
    result = calc.estimate_yield(crop, area_acres, price_per_kg)
    return result.model_dump()


def _convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
) -> float:
    """Convert between agricultural units."""
    conversions = {
        ("kg", "quintal"): lambda x: x / 100,
        ("quintal", "kg"): lambda x: x * 100,
        ("kg", "ton"): lambda x: x / 1000,
        ("ton", "kg"): lambda x: x * 1000,
        ("acre", "hectare"): lambda x: x * 0.4047,
        ("hectare", "acre"): lambda x: x / 0.4047,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        return round(conversions[key](value), 4)
    
    return value  # No conversion available


# Auto-register on module import
try:
    registry = get_tool_registry()
    
    registry.add_tool(
        _calculate_aisp,
        name="calculate_aisp",
        description="Calculate All-Inclusive Sourcing Price (AISP) for a transaction. Returns farmer payout, logistics, handling, platform fee, and total buyer cost.",
        category="calculator",
    )
    
    registry.add_tool(
        _estimate_yield,
        name="estimate_yield",
        description="Estimate crop yield for a given area in acres. Returns expected production in kg and revenue estimate if price is provided.",
        category="calculator",
    )
    
    registry.add_tool(
        _convert_units,
        name="convert_units",
        description="Convert between agricultural units (kg/quintal/ton, acre/hectare).",
        category="calculator",
    )
except Exception as e:
    logger.debug(f"Calculator tool registration deferred: {e}")
