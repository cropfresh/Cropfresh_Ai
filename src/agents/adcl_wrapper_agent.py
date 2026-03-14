"""
ADCL Wrapper Agent
==================
Bridges the standalone ADCLAgent engine into the BaseAgent
interface so the Supervisor can route 'crop recommendation'
and 'what to sow' queries to the ADCL pipeline.

The ADCLAgent (src/agents/adcl/engine.py) does NOT inherit
BaseAgent — this lightweight wrapper adapts its API.

Author: CropFresh AI Team
Version: 2.1.0
"""

from typing import Any, Optional

from loguru import logger

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.prompt_context import build_system_prompt
from src.memory.state_manager import AgentExecutionState

# * District extraction keywords (Karnataka focus)
DISTRICTS = [
    "bangalore", "bengaluru", "kolar", "mysore", "mysuru",
    "tumkur", "tumakuru", "mandya", "hassan", "shimoga",
    "shivamogga", "davangere", "bellary", "ballari",
    "raichur", "hubli", "dharwad", "belgaum", "belagavi",
    "udupi", "mangalore", "dakshina kannada", "chikkaballapur",
]


class ADCLWrapperAgent(BaseAgent):
    """
    Wrapper for ADCL engine — routes demand/sowing queries
    to generate_weekly_report() and formats the output.

    Usage via Supervisor:
        "What should I sow this season?"
        "Which crops have highest demand?"
        "Weekly crop recommendation for Kolar"
    """

    def __init__(self, llm: Any = None, **kwargs: Any) -> None:
        config = AgentConfig(
            name="adcl_agent",
            description=(
                "Recommends crops to sow based on demand signals, "
                "seasonality, and price forecasts"
            ),
            temperature=0.5,
            max_tokens=600,
        )
        super().__init__(config=config, llm=llm, **kwargs)

        # * Lazy-init the engine to avoid import errors at module level
        self._engine = None

    async def initialize(self) -> bool:
        """Create the underlying ADCL engine."""
        try:
            from src.agents.adcl.engine import get_adcl_agent
            self._engine = get_adcl_agent(llm=self.llm)
            self._initialized = True
            logger.info("ADCLWrapperAgent initialized")
        except Exception as exc:
            logger.warning("ADCL engine init failed: {}", exc)
            self._initialized = True  # * Don't block startup
        return True

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        return build_system_prompt(
            role_description="You are CropFresh's Crop Recommendation Agent.",
            domain_prompt=(
                "Help farmers decide what to sow based on market demand, "
                "seasonality, and price forecasts for Karnataka. "
                "Use the ADCL (Adaptive Demand-driven Crop Lifecycle) engine "
                "to generate data-backed recommendations. "
                "Always specify the district, top 3-5 crops, demand scores, "
                "and reasoning for each recommendation."
            ),
            context=context,
        )

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """
        Generate crop recommendations from the ADCL engine.

        Falls back to LLM-only advice if the engine isn't available.
        """
        district = self._extract_district(query, context)

        # * Try the ADCL engine first
        if self._engine:
            try:
                report = await self._engine.generate_weekly_report(
                    district=district,
                )
                content = self._format_report(report, district)
                return AgentResponse(
                    content=content,
                    agent_name=self.name,
                    confidence=0.85,
                    steps=["adcl_report_generated"],
                    suggested_actions=[
                        "Check current market prices",
                        "Get detailed growing guide for top crop",
                    ],
                )
            except Exception as exc:
                logger.warning("ADCL engine failed, using LLM fallback: {}", exc)

        # * Fallback: LLM-only advice
        if self.llm:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": query},
            ]
            answer = await self.generate_with_llm(messages)
            return AgentResponse(
                content=answer,
                agent_name=self.name,
                confidence=0.6,
                steps=["llm_fallback"],
            )

        return AgentResponse(
            content=(
                "I can recommend crops to sow based on demand and season. "
                "Please try again when the service is fully available."
            ),
            agent_name=self.name,
            confidence=0.3,
            steps=["no_engine_no_llm"],
        )

    # ─────────────────────────────────────────────────────────
    # * Helpers
    # ─────────────────────────────────────────────────────────

    def _extract_district(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> str:
        """Extract district from query text or user context."""
        # * Check context first
        if context:
            profile = context.get("user_profile", {})
            if profile.get("district"):
                return profile["district"]

        # * Keyword match in query
        query_lower = query.lower()
        for district in DISTRICTS:
            if district in query_lower:
                return district.title()

        return "Bangalore"  # * Default Karnataka metro

    def _format_report(self, report: Any, district: str) -> str:
        """Format ADCL WeeklyReport into a readable response."""
        lines = [f"📊 **Weekly Crop Recommendations — {district}**\n"]

        if not hasattr(report, "crops") or not report.crops:
            lines.append("No crop recommendations available this week.")
            return "\n".join(lines)

        for i, crop in enumerate(report.crops[:5], 1):
            label = getattr(crop, "label", "")
            emoji = "🟢" if label == "green" else "🟡" if label == "yellow" else "🔴"
            name = getattr(crop, "commodity", "Unknown")
            score = getattr(crop, "demand_score", 0)
            summary = getattr(crop, "summary", "")

            lines.append(f"{emoji} **{i}. {name}** (Score: {score:.1f})")
            if summary:
                lines.append(f"   {summary}")

        if hasattr(report, "generated_at"):
            lines.append(f"\n_Generated: {report.generated_at}_")

        return "\n".join(lines)
