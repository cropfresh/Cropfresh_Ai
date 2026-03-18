"""
Logistics Wrapper Agent
=======================
Bridges the standalone LogisticsRouter engine into the BaseAgent
interface so the Supervisor can route delivery/transport queries.

The LogisticsRouter (src/agents/logistics_router/engine.py) does NOT
inherit BaseAgent — this lightweight wrapper adapts its API.

Author: CropFresh AI Team
Version: 2.1.0
"""

from typing import Any, Optional

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.prompt_context import build_system_prompt
from src.memory.state_manager import AgentExecutionState


class LogisticsWrapperAgent(BaseAgent):
    """
    Wrapper for LogisticsRouter — routes delivery/transport queries
    to the route optimizer and formats cost estimates.

    Usage via Supervisor:
        "How much will delivery cost for 500kg to Bangalore?"
        "Plan route for my produce pickup"
        "What vehicle do I need for 2 tons?"
    """

    def __init__(self, llm: Any = None, **kwargs: Any) -> None:
        config = AgentConfig(
            name="logistics_agent",
            description=(
                "Optimizes delivery routes, calculates transport costs, "
                "and assigns vehicles for produce logistics"
            ),
            temperature=0.3,
            max_tokens=400,
        )
        super().__init__(config=config, llm=llm, **kwargs)
        self._router = None

    async def initialize(self) -> bool:
        """Create the underlying LogisticsRouter engine."""
        try:
            from src.agents.logistics_router.engine import get_logistics_router

            self._router = get_logistics_router()
            self._initialized = True
            logger.info("LogisticsWrapperAgent initialized")
        except Exception as exc:
            logger.warning("LogisticsRouter init failed: {}", exc)
            self._initialized = True  # * Don't block startup
        return True

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        return build_system_prompt(
            role_description="You are CropFresh's Logistics Agent.",
            domain_prompt=(
                "Help farmers and buyers understand delivery costs, "
                "optimal routes, and vehicle requirements for produce "
                "transport in Karnataka. CropFresh targets delivery "
                "cost < ₹2.5/kg. Optimize for vehicle utilization, "
                "cold-chain requirements, and multi-stop clustering."
            ),
            context=context,
            agent_domain="logistics",
        )

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Handle logistics queries.

        For structured requests (with pickup/delivery data in context),
        delegates to the engine. For natural language, uses LLM.
        """
        # * Check if context has structured pickup/delivery data
        if context and context.get("pickups") and context.get("delivery"):
            return await self._handle_structured_route(context)

        # * Natural language fallback
        if self.llm:
            messages = [
                {"role": "system", "content": self._get_system_prompt(context)},
                {"role": "user", "content": query},
            ]
            answer = await self.generate_with_llm(messages, context=context)
            return AgentResponse(
                content=answer,
                agent_name=self.name,
                confidence=0.7,
                steps=["llm_logistics_advice"],
            )

        return AgentResponse(
            content=(
                "I can help with delivery routing and cost estimates. "
                "Please provide pickup and delivery locations, "
                "or ask about general logistics costs."
            ),
            agent_name=self.name,
            confidence=0.3,
            steps=["no_engine_no_llm"],
        )

    async def _handle_structured_route(
        self,
        context: dict,
    ) -> AgentResponse:
        """Run the actual route optimizer with structured data."""
        if not self._router:
            return AgentResponse(
                content="Logistics routing engine is not available.",
                agent_name=self.name,
                confidence=0.2,
            )

        try:
            from src.agents.logistics_router.models import (
                DeliveryPoint,
                PickupPoint,
            )

            pickups = [PickupPoint(**p) for p in context["pickups"]]
            delivery = DeliveryPoint(**context["delivery"])

            result = await self._router.plan_route(
                pickups=pickups,
                delivery=delivery,
            )

            if not result:
                return AgentResponse(
                    content="Could not plan a route with the given points.",
                    agent_name=self.name,
                    confidence=0.5,
                )

            content = self._format_route(result)
            return AgentResponse(
                content=content,
                agent_name=self.name,
                confidence=0.9,
                steps=["route_optimized"],
            )

        except Exception as exc:
            logger.error("Route planning failed: {}", exc)
            return AgentResponse(
                content=f"Route planning failed: {exc}",
                agent_name=self.name,
                confidence=0.2,
                error=str(exc),
            )

    def _format_route(self, result: Any) -> str:
        """Format RouteResult into readable response."""
        return (
            f"🚚 **Route Plan**\n\n"
            f"• Vehicle: {result.vehicle_type}\n"
            f"• Stops: {result.cluster_size}\n"
            f"• Distance: {result.total_distance_km:.1f} km\n"
            f"• Weight: {result.total_weight_kg:.0f} kg\n"
            f"• Cost: ₹{result.estimated_cost:.0f} "
            f"(₹{result.cost_per_kg:.2f}/kg)\n"
            f"• Utilization: {result.utilization_pct:.0f}%\n"
            f"• Est. Duration: {result.estimated_duration_hours:.1f} hours"
        )
