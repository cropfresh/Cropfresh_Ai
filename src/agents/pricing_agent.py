"""
Pricing Agent
=============
Calculates fair prices and provides market recommendations.

Uses:
- Agmarknet API for current market prices
- Historical trend analysis
- Weather data for supply prediction
- AISP calculation
"""

from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from src.orchestrator.llm_provider import BaseLLMProvider, LLMMessage
from src.tools.agmarknet import AgmarknetTool, AgmarknetPrice


class PriceRecommendation(BaseModel):
    """Price recommendation result."""
    
    commodity: str
    location: str
    current_price: float  # ₹/kg
    current_price_quintal: float  # ₹/quintal
    
    # Market context
    market_min: float
    market_max: float
    
    # Recommendation
    recommended_action: str  # "sell", "hold", "wait"
    confidence: float
    reason: str
    
    # AISP (for buyers)
    aisp_per_kg: Optional[float] = None
    aisp_breakdown: Optional[dict] = None
    
    # Metadata
    data_source: str
    timestamp: datetime


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


class PricingAgent:
    """
    Dynamic Pricing Engine (DPLE).
    
    Provides:
    - Real-time market prices
    - Price trend analysis
    - Sell/hold recommendations
    - AISP calculations for buyers
    
    Usage:
        agent = PricingAgent(llm=provider, agmarknet_api_key="key")
        rec = await agent.get_recommendation("Tomato", "Kolar", quantity_kg=200)
    """
    
    DEADHEAD_FACTOR_TABLE = [
        (80, 101, 0.00),
        (60, 80, 0.10),
        (40, 60, 0.20),
        (0, 40, 0.35),
    ]

    LOGISTICS_RATE_TABLE = [
        (0, 15, 1.5),
        (15, 50, 1.2),
        (50, 100, 1.0),
        (100, 1000000, 0.8),
    ]

    PLATFORM_FEE_TIERS = [
        (1000, float("inf"), 0.05),
        (500, 1000, 0.06),
        (100, 500, 0.07),
        (0, 100, 0.08),
    ]

    RISK_BUFFER_PCT = 0.02
    MANDI_CAP_MULTIPLIER = 1.05
    COLD_CHAIN_PREMIUM_PER_KM = 0.5
    
    def __init__(
        self,
        llm: Optional[BaseLLMProvider] = None,
        agmarknet_api_key: str = "",
        use_mock: bool = True,
    ):
        """
        Initialize Pricing Agent.
        
        Args:
            llm: LLM provider for reasoning
            agmarknet_api_key: API key for Agmarknet
            use_mock: Use mock data if API unavailable
        """
        self.llm = llm
        self.agmarknet = AgmarknetTool(api_key=agmarknet_api_key)
        self.use_mock = use_mock
    
    async def get_current_price(
        self,
        commodity: str,
        state: str = "Karnataka",
        district: Optional[str] = None,
    ) -> list[AgmarknetPrice]:
        """Get current market prices."""
        
        if self.use_mock or not self.agmarknet.api_key:
            logger.info(f"Using mock prices for {commodity}")
            return self.agmarknet.get_mock_prices(commodity, state, district or "Kolar")
        
        return await self.agmarknet.get_prices(commodity, state, district)
    
    async def get_recommendation(
        self,
        commodity: str,
        location: str,
        quantity_kg: float = 100,
        asking_price: Optional[float] = None,
    ) -> PriceRecommendation:
        """
        Get sell/hold recommendation for a commodity.
        
        Args:
            commodity: Crop name (e.g., "Tomato")
            location: District/market name
            quantity_kg: Quantity in kg
            asking_price: Farmer's asking price (optional)
            
        Returns:
            PriceRecommendation with action and reasoning
        """
        # 1. Get current prices
        prices = await self.get_current_price(commodity, district=location)
        
        if not prices:
            return PriceRecommendation(
                commodity=commodity,
                location=location,
                current_price=0,
                current_price_quintal=0,
                market_min=0,
                market_max=0,
                recommended_action="unknown",
                confidence=0,
                reason="Could not fetch market prices",
                data_source="none",
                timestamp=datetime.now(),
            )
        
        price = prices[0]
        price_per_kg = price.modal_price_per_kg
        
        # 2. Analyze (simple heuristics for now, LLM for complex analysis)
        action, confidence, reason = self._analyze_simple(
            price, quantity_kg, asking_price
        )
        
        aisp = self.calculate_aisp(
            farmer_price_per_kg=price_per_kg,
            quantity_kg=quantity_kg,
            distance_km=30,
            mandi_modal_per_kg=price.modal_price_per_kg,
        )
        
        return PriceRecommendation(
            commodity=commodity,
            location=location,
            current_price=price_per_kg,
            current_price_quintal=price.modal_price,
            market_min=price.min_price / 100,
            market_max=price.max_price / 100,
            recommended_action=action,
            confidence=confidence,
            reason=reason,
            aisp_per_kg=aisp.aisp_per_kg,
            aisp_breakdown=aisp.model_dump(),
            data_source="agmarknet" if not self.use_mock else "mock",
            timestamp=datetime.now(),
        )
    
    def _analyze_simple(
        self,
        price: AgmarknetPrice,
        quantity_kg: float,
        asking_price: Optional[float],
    ) -> tuple[str, float, str]:
        """
        Simple rule-based analysis.
        
        Returns: (action, confidence, reason)
        """
        modal = price.modal_price_per_kg
        min_price = price.min_price / 100
        max_price = price.max_price / 100
        
        # Price position in market range
        range_width = max_price - min_price if max_price > min_price else 1
        position = (modal - min_price) / range_width
        
        # If price is in upper 30% of range, recommend sell
        if position > 0.7:
            return (
                "sell",
                0.85,
                f"Current price (₹{modal:.1f}/kg) is near market high (₹{max_price:.1f}/kg). Good time to sell."
            )
        
        # If price is in lower 30% of range, recommend hold
        elif position < 0.3:
            return (
                "hold",
                0.75,
                f"Current price (₹{modal:.1f}/kg) is near market low. Consider waiting for better prices."
            )
        
        # Middle range - depends on quantity
        else:
            if quantity_kg > 500:
                return (
                    "sell",
                    0.65,
                    f"Price is moderate (₹{modal:.1f}/kg). With {quantity_kg}kg, selling now avoids spoilage risk."
                )
            else:
                return (
                    "hold",
                    0.55,
                    f"Price is moderate (₹{modal:.1f}/kg). Small quantity ({quantity_kg}kg) can wait for better prices."
                )
    
    def calculate_aisp(
        self,
        farmer_price_per_kg: float,
        quantity_kg: float,
        distance_km: float = 30,
        handling_per_kg: float = 0.5,
        mandi_modal_per_kg: Optional[float] = None,
        route_utilization_pct: float = 50.0,
        cold_chain: bool = False,
    ) -> AISPCalculation:
        """
        Calculate business-aligned AISP:
        Farmer_Payout + Logistics + Deadhead + Handling + Platform_Fee + Risk_Buffer.
        """
        if quantity_kg <= 0:
            raise ValueError("quantity_kg must be greater than 0")
        if farmer_price_per_kg < 0:
            raise ValueError("farmer_price_per_kg cannot be negative")
        if distance_km < 0:
            raise ValueError("distance_km cannot be negative")
        if not 0 <= route_utilization_pct <= 100:
            raise ValueError("route_utilization_pct must be between 0 and 100")

        farmer_payout = farmer_price_per_kg * quantity_kg

        logistics_rate = self._get_logistics_rate(distance_km)
        logistics_cost = distance_km * logistics_rate * quantity_kg / 1000
        if cold_chain:
            logistics_cost += distance_km * self.COLD_CHAIN_PREMIUM_PER_KM

        deadhead_factor = self._get_deadhead_factor(route_utilization_pct)
        deadhead_surcharge = logistics_cost * deadhead_factor

        handling_cost = handling_per_kg * quantity_kg

        subtotal = farmer_payout + logistics_cost + deadhead_surcharge + handling_cost
        platform_fee_pct = self._get_platform_fee(quantity_kg)
        platform_fee = subtotal * platform_fee_pct

        risk_buffer = subtotal * self.RISK_BUFFER_PCT

        total_aisp = subtotal + platform_fee + risk_buffer
        aisp_per_kg = total_aisp / quantity_kg

        mandi_cap_applied = False
        if mandi_modal_per_kg:
            mandi_cap = mandi_modal_per_kg * self.MANDI_CAP_MULTIPLIER
            if aisp_per_kg > mandi_cap:
                aisp_per_kg = mandi_cap
                total_aisp = mandi_cap * quantity_kg
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
            risk_buffer_pct=self.RISK_BUFFER_PCT,
            total_aisp=round(total_aisp, 2),
            aisp_per_kg=round(aisp_per_kg, 2),
            mandi_cap_applied=mandi_cap_applied,
        )
    
    def _get_logistics_rate(self, distance_km: float) -> float:
        """Get logistics rate (₹/km/kg) based on distance."""
        for min_d, max_d, rate in self.LOGISTICS_RATE_TABLE:
            if min_d <= distance_km < max_d:
                return rate
        return self.LOGISTICS_RATE_TABLE[-1][2]

    def _get_deadhead_factor(self, route_utilization_pct: float) -> float:
        """Get deadhead surcharge factor based on route utilization."""
        for min_u, max_u, factor in self.DEADHEAD_FACTOR_TABLE:
            if min_u <= route_utilization_pct < max_u:
                return factor
        return self.DEADHEAD_FACTOR_TABLE[-1][2]
    
    def _get_platform_fee(self, quantity_kg: float) -> float:
        """Get platform fee percentage based on quantity."""
        if quantity_kg > 1000:
            return 0.05
        if quantity_kg >= 500:
            return 0.06
        if quantity_kg >= 100:
            return 0.07
        return 0.08

    async def get_price_trend(
        self,
        commodity: str,
        district: str = "Bangalore",
        days: int = 30,
    ) -> dict[str, float | str]:
        """
        Analyze trend using historical mandi prices.

        Returns: trend, volatility_index, 7d_avg, 30d_avg, recommendation.
        """
        if days <= 0:
            raise ValueError("days must be greater than 0")

        history = await self.agmarknet.get_historical_prices(
            commodity=commodity,
            state="Karnataka",
            district=district,
            days=max(days, 30),
        )
        if not history:
            return {
                "trend": "stable",
                "volatility_index": 0.0,
                "7d_avg": 0.0,
                "30d_avg": 0.0,
                "recommendation": "hold_3_days",
            }

        sorted_history = sorted(history, key=lambda price: price.date)
        prices_per_kg = [price.modal_price_per_kg for price in sorted_history[-days:]]
        seven_day_window = prices_per_kg[-7:] if len(prices_per_kg) >= 7 else prices_per_kg
        thirty_day_window = prices_per_kg[-30:] if len(prices_per_kg) >= 30 else prices_per_kg
        avg_7d = mean(seven_day_window)
        avg_30d = mean(thirty_day_window)
        volatility = pstdev(prices_per_kg) / avg_30d if avg_30d > 0 else 0.0
        volatility_index = min(max(volatility, 0.0), 1.0)

        if avg_7d > avg_30d * 1.03:
            trend = "rising"
            recommendation = "sell_now"
        elif avg_7d < avg_30d * 0.97:
            trend = "falling"
            recommendation = "hold_7_days"
        else:
            trend = "stable"
            recommendation = "hold_3_days"

        return {
            "trend": trend,
            "volatility_index": round(volatility_index, 4),
            "7d_avg": round(avg_7d, 2),
            "30d_avg": round(avg_30d, 2),
            "recommendation": recommendation,
        }

    def get_seasonal_adjustment(self, commodity: str, month: int) -> float:
        """Return seasonal multiplier for major Karnataka crops."""
        if month < 1 or month > 12:
            raise ValueError("month must be between 1 and 12")

        seasonal_factors = {
            "tomato": {5: 1.30, 6: 1.20, 7: 1.15, 11: 0.90, 12: 0.85},
            "onion": {11: 0.80, 12: 0.82, 1: 0.88, 5: 1.10, 6: 1.15},
            "potato": {2: 0.90, 3: 0.88, 9: 1.08, 10: 1.12},
            "cabbage": {12: 0.85, 1: 0.86, 4: 1.10, 5: 1.12},
            "cauliflower": {12: 0.86, 1: 0.88, 4: 1.10, 5: 1.15},
        }

        commodity_key = commodity.strip().lower()
        commodity_factors = seasonal_factors.get(commodity_key, {})
        return commodity_factors.get(month, 1.0)
    
    async def get_price_with_llm_analysis(
        self,
        commodity: str,
        location: str,
        context: str = "",
    ) -> str:
        """
        Get price recommendation with LLM reasoning.
        
        Uses the LLM to provide natural language analysis.
        """
        if not self.llm:
            rec = await self.get_recommendation(commodity, location)
            return f"Current {commodity} price in {location}: ₹{rec.current_price:.1f}/kg. {rec.reason}"
        
        # Get market data
        prices = await self.get_current_price(commodity, district=location)
        price = prices[0] if prices else None
        
        if not price:
            return f"Sorry, I couldn't fetch current prices for {commodity} in {location}."
        
        # Build prompt for LLM
        messages = [
            LLMMessage(
                role="system",
                content="""You are CropFresh AI's Pricing Agent. Analyze market prices and provide recommendations.

Be concise, specific, and actionable. Include:
1. Current price with source
2. Whether to sell now or wait
3. Brief reasoning (weather, trends, etc.)

Use ₹ symbol for prices. Respond in 2-3 sentences max."""
            ),
            LLMMessage(
                role="user",
                content=f"""Commodity: {commodity}
Location: {location}
Current Modal Price: ₹{price.modal_price:.0f}/quintal (₹{price.modal_price_per_kg:.1f}/kg)
Min Price: ₹{price.min_price:.0f}/quintal
Max Price: ₹{price.max_price:.0f}/quintal
Date: {price.date.strftime('%d %b %Y')}
{f'Additional context: {context}' if context else ''}

Should the farmer sell now or wait?"""
            ),
        ]
        
        response = await self.llm.generate(messages, temperature=0.7, max_tokens=200)
        return response.content
