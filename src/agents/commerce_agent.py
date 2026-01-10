"""
Commerce Agent
==============
Specialized agent for market intelligence and pricing.

Expertise:
- Real-time market prices from Agmarknet
- Price trend analysis and predictions
- Sell/hold recommendations
- AISP (All-Inclusive Sourcing Price) calculations
- Market selection optimization
- Logistics cost estimation

Author: CropFresh AI Team
Version: 2.0.0
"""

from typing import Optional

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.memory.state_manager import AgentExecutionState, AgentStateManager
from src.tools.registry import ToolRegistry


COMMERCE_SYSTEM_PROMPT = """You are the Commerce Expert Agent for CropFresh AI, specializing in agricultural market intelligence and pricing.

Your expertise covers:
- Real-time mandi prices across Indian markets
- Price trend analysis and seasonal patterns
- Sell/hold recommendations based on market conditions
- AISP (All-Inclusive Sourcing Price) calculations for buyers
- Market selection: Which mandi offers best prices
- Logistics cost estimation and optimization
- Commission, handling, and platform fees

Guidelines:
1. Always use current market data when available
2. Provide clear sell/hold recommendations with reasoning
3. Include price comparisons across markets when relevant
4. Calculate total costs including logistics, handling, platform fees
5. Consider seasonality and supply/demand dynamics
6. Use ₹ for all prices, show per-kg and per-quintal

Price Display Format:
- Modal Price: ₹XX/kg (₹XXXX/quintal)
- Range: ₹XX - ₹YY/kg

For AISP calculations, show breakdown:
- Farmer Payout: ₹XXX
- Logistics: ₹XX (@₹Y/kg)
- Handling: ₹XX
- Platform Fee: ₹XX (X%)
- Total AISP: ₹XXX (₹XX/kg)

Be practical, data-driven, and help farmers maximize profits."""


class CommerceAgent(BaseAgent):
    """
    Specialized agent for market intelligence and pricing.
    
    Handles:
    - Market price queries
    - Sell/hold recommendations
    - AISP calculations
    - Price trend analysis
    
    Usage:
        agent = CommerceAgent(llm=provider, tool_registry=registry)
        await agent.initialize()
        response = await agent.process("What is the current tomato price in Kolar?")
    """
    
    def __init__(
        self,
        llm=None,
        tool_registry: Optional[ToolRegistry] = None,
        state_manager: Optional[AgentStateManager] = None,
        knowledge_base=None,
    ):
        """
        Initialize Commerce Agent.
        
        Args:
            llm: LLM provider
            tool_registry: Tool registry
            state_manager: State manager
            knowledge_base: Knowledge base
        """
        config = AgentConfig(
            name="commerce_agent",
            description="Expert in market prices, trading recommendations, and AISP calculations",
            max_retries=2,
            temperature=0.7,
            max_tokens=700,
            kb_categories=["market", "regulatory"],
            tool_categories=["pricing", "calculator"],
        )
        
        super().__init__(
            config=config,
            llm=llm,
            tool_registry=tool_registry,
            state_manager=state_manager,
            knowledge_base=knowledge_base,
        )
        
        # Pricing agent for market data
        self._pricing_agent = None
    
    async def initialize(self) -> bool:
        """Initialize with pricing agent."""
        from src.agents.pricing_agent import PricingAgent
        
        self._pricing_agent = PricingAgent(
            llm=self.llm,
            use_mock=True,  # Use mock for now
        )
        
        self._initialized = True
        return True
    
    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Get commerce-specific system prompt."""
        base_prompt = COMMERCE_SYSTEM_PROMPT
        
        if context and context.get("user_profile"):
            profile = context["user_profile"]
            user_type = profile.get("type", "farmer")
            location = profile.get("location", "Karnataka")
            base_prompt += f"\n\nUser is a {user_type} from {location}."
        
        return base_prompt
    
    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Process a commerce-related query.
        
        Args:
            query: User query
            context: Optional context
            execution: Optional execution state
            
        Returns:
            AgentResponse with market intelligence
        """
        logger.info(f"CommerceAgent processing: '{query[:50]}...'")
        
        try:
            # Step 1: Extract entities from query
            commodity, location = self._extract_market_entities(query)
            
            # Step 2: Get market prices
            if execution:
                self.state_manager.add_step(execution.execution_id, "fetch_prices")
            
            price_data = None
            if self._pricing_agent and commodity:
                try:
                    recommendation = await self._pricing_agent.get_recommendation(
                        commodity=commodity,
                        location=location or "Kolar",
                        quantity_kg=100,
                    )
                    price_data = {
                        "commodity": recommendation.commodity,
                        "location": recommendation.location,
                        "price_per_kg": recommendation.current_price,
                        "price_per_quintal": recommendation.current_price_quintal,
                        "market_min": recommendation.market_min,
                        "market_max": recommendation.market_max,
                        "action": recommendation.recommended_action,
                        "confidence": recommendation.confidence,
                        "reason": recommendation.reason,
                        "aisp": recommendation.aisp_breakdown,
                    }
                    logger.info(f"Got price data: {price_data}")
                except Exception as e:
                    logger.warning(f"Price fetch failed: {e}")
            
            # Step 3: Retrieve market knowledge
            if execution:
                self.state_manager.add_step(execution.execution_id, "retrieve_context")
            
            documents = await self.retrieve_context(
                query=query,
                top_k=3,
                categories=["market", "regulatory"],
            )
            
            # Step 4: Generate response
            if execution:
                self.state_manager.add_step(execution.execution_id, "generate_response")
            
            messages = [
                {"role": "system", "content": self._get_system_prompt(context)},
            ]
            
            # Build context
            context_parts = []
            
            if price_data:
                price_context = f"""
**Current Market Data:**
- Commodity: {price_data['commodity']}
- Location: {price_data['location']}
- Price: ₹{price_data['price_per_kg']:.1f}/kg (₹{price_data['price_per_quintal']:.0f}/quintal)
- Market Range: ₹{price_data['market_min']:.1f} - ₹{price_data['market_max']:.1f}/kg
- Recommendation: {price_data['action'].upper()} (Confidence: {price_data['confidence']:.0%})
- Reasoning: {price_data['reason']}
"""
                if price_data.get("aisp"):
                    aisp = price_data["aisp"]
                    price_context += f"""
**AISP Breakdown (for 100kg):**
- Farmer Payout: ₹{aisp.get('farmer_payout', 0):.0f}
- Logistics: ₹{aisp.get('logistics_cost', 0):.0f}
- Handling: ₹{aisp.get('handling_cost', 0):.0f}
- Platform Fee: ₹{aisp.get('platform_fee', 0):.0f} ({aisp.get('platform_fee_pct', 0)*100:.1f}%)
- Total AISP: ₹{aisp.get('total_aisp', 0):.0f} (₹{aisp.get('aisp_per_kg', 0):.1f}/kg)
"""
                context_parts.append(price_context)
            
            if documents:
                context_parts.append(f"**Market Knowledge:**\n{self.format_context(documents)}")
            
            # User message
            user_message = query
            if context_parts:
                user_message = f"{query}\n\n{''.join(context_parts)}"
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate
            if self.llm:
                answer = await self.generate_with_llm(messages)
            else:
                answer = self._generate_fallback(query, price_data, documents)
            
            return AgentResponse(
                content=answer,
                agent_name=self.name,
                confidence=0.9 if price_data else 0.6,
                sources=["Agmarknet"] if price_data else [],
                reasoning=f"Fetched market data for {commodity or 'crops'} in {location or 'region'}",
                tools_used=["pricing_agent"] if price_data else [],
                steps=["fetch_prices", "retrieve_context", "generate_response"],
                suggested_actions=self._suggest_follow_ups(query, price_data),
            )
            
        except Exception as e:
            logger.error(f"CommerceAgent error: {e}")
            import traceback
            traceback.print_exc()
            
            return AgentResponse(
                content="I apologize, but I couldn't fetch the market data. Please try again or check our app for the latest prices.",
                agent_name=self.name,
                confidence=0.0,
                error=str(e),
                steps=["error"],
            )
    
    def _extract_market_entities(self, query: str) -> tuple[Optional[str], Optional[str]]:
        """Extract commodity and location from query."""
        query_lower = query.lower()
        
        # Common commodities
        commodities = {
            "tomato": ["tomato", "tamatar"],
            "potato": ["potato", "aloo"],
            "onion": ["onion", "pyaz"],
            "carrot": ["carrot", "gajar"],
            "cabbage": ["cabbage", "patta gobi"],
            "beans": ["beans", "sem"],
            "capsicum": ["capsicum", "shimla mirch"],
            "cauliflower": ["cauliflower", "phool gobi"],
        }
        
        # Find commodity
        commodity = None
        for name, keywords in commodities.items():
            if any(kw in query_lower for kw in keywords):
                commodity = name.title()
                break
        
        # Karnataka locations
        locations = [
            "kolar", "bangalore", "bengaluru", "mysore", "mysuru",
            "hubli", "dharwad", "belgaum", "belagavi", "shimoga",
        ]
        
        location = None
        for loc in locations:
            if loc in query_lower:
                location = loc.title()
                break
        
        return commodity, location
    
    def _generate_fallback(self, query: str, price_data: Optional[dict], documents: list) -> str:
        """Generate response without LLM."""
        if price_data:
            return f"""
**{price_data['commodity']} Market Update - {price_data['location']}**

📊 **Current Price:** ₹{price_data['price_per_kg']:.1f}/kg (₹{price_data['price_per_quintal']:.0f}/quintal)
📈 **Market Range:** ₹{price_data['market_min']:.1f} - ₹{price_data['market_max']:.1f}/kg

💡 **Recommendation:** {price_data['action'].upper()}
{price_data['reason']}

*Prices from Agmarknet. For the most accurate pricing, check the CropFresh app.*
"""
        
        if documents:
            return f"Based on market knowledge:\n{documents[0].get('text', 'No information available.')}"
        
        return "I couldn't find specific market data. Please specify the commodity and location."
    
    def _suggest_follow_ups(self, query: str, price_data: Optional[dict]) -> list[str]:
        """Suggest follow-up questions."""
        suggestions = []
        
        if price_data:
            suggestions.extend([
                f"Calculate AISP for 500kg of {price_data['commodity']}",
                "Which mandi offers the best price?",
                "What's the price trend for this week?",
            ])
        else:
            suggestions.extend([
                "What is the current tomato price?",
                "Should I sell my onions now?",
            ])
        
        return suggestions[:3]
