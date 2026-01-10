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
    handling_cost: float
    platform_fee: float
    platform_fee_pct: float
    
    total_aisp: float
    aisp_per_kg: float


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
    
    # Platform fee tiers
    PLATFORM_FEE_TIERS = [
        (0, 100, 0.08),      # 0-100 kg: 8%
        (100, 500, 0.06),    # 100-500 kg: 6%  
        (500, float("inf"), 0.04),  # 500+ kg: 4%
    ]
    
    # Logistics rates (₹/kg based on distance)
    LOGISTICS_RATES = [
        (0, 25, 2.0),    # 0-25 km: ₹2/kg
        (25, 50, 2.5),   # 25-50 km: ₹2.5/kg
        (50, 100, 3.0),  # 50-100 km: ₹3/kg
        (100, float("inf"), 4.0),  # 100+ km: ₹4/kg
    ]
    
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
        
        # 3. Calculate AISP if buyer needs it
        aisp = self.calculate_aisp(
            farmer_price_per_kg=price_per_kg,
            quantity_kg=quantity_kg,
            distance_km=30,  # Default estimate
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
    ) -> AISPCalculation:
        """
        Calculate All-Inclusive Sourcing Price.
        
        AISP = Farmer_Payout + Logistics + Handling + Platform_Fee
        
        Args:
            farmer_price_per_kg: Price paid to farmer (₹/kg)
            quantity_kg: Quantity in kg
            distance_km: Distance from farm to buyer
            handling_per_kg: Handling cost per kg
            
        Returns:
            AISPCalculation with full breakdown
        """
        # Farmer payout
        farmer_payout = farmer_price_per_kg * quantity_kg
        
        # Logistics cost
        logistics_rate = self._get_logistics_rate(distance_km)
        logistics_cost = logistics_rate * quantity_kg
        
        # Handling cost
        handling_cost = handling_per_kg * quantity_kg
        
        # Platform fee (on farmer_payout + logistics + handling)
        subtotal = farmer_payout + logistics_cost + handling_cost
        platform_fee_pct = self._get_platform_fee(quantity_kg)
        platform_fee = subtotal * platform_fee_pct
        
        # Total AISP
        total_aisp = subtotal + platform_fee
        
        return AISPCalculation(
            farmer_price_per_kg=farmer_price_per_kg,
            quantity_kg=quantity_kg,
            farmer_payout=farmer_payout,
            logistics_cost=logistics_cost,
            handling_cost=handling_cost,
            platform_fee=platform_fee,
            platform_fee_pct=platform_fee_pct,
            total_aisp=total_aisp,
            aisp_per_kg=total_aisp / quantity_kg,
        )
    
    def _get_logistics_rate(self, distance_km: float) -> float:
        """Get logistics rate based on distance."""
        for min_d, max_d, rate in self.LOGISTICS_RATES:
            if min_d <= distance_km < max_d:
                return rate
        return self.LOGISTICS_RATES[-1][2]
    
    def _get_platform_fee(self, quantity_kg: float) -> float:
        """Get platform fee percentage based on quantity."""
        for min_q, max_q, fee in self.PLATFORM_FEE_TIERS:
            if min_q <= quantity_kg < max_q:
                return fee
        return self.PLATFORM_FEE_TIERS[-1][2]
    
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
