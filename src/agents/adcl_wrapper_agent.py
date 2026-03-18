"""
ADCL wrapper agent for supervisor compatibility.
"""

from typing import Any, Optional

from loguru import logger

from src.agents.adcl.presentation import format_weekly_report
from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.prompt_context import build_system_prompt
from src.memory.state_manager import AgentExecutionState

DISTRICTS = [
    "bangalore",
    "bengaluru",
    "kolar",
    "mysore",
    "mysuru",
    "tumkur",
    "tumakuru",
    "mandya",
    "hassan",
    "shimoga",
    "shivamogga",
    "davangere",
    "bellary",
    "ballari",
    "raichur",
    "hubli",
    "dharwad",
    "belgaum",
    "belagavi",
    "udupi",
    "mangalore",
    "dakshina kannada",
    "chikkaballapur",
]


class ADCLWrapperAgent(BaseAgent):
    """Wrap the canonical ADCL service in the BaseAgent interface."""

    def __init__(self, llm: Any = None, **kwargs: Any) -> None:
        config = AgentConfig(
            name="adcl_agent",
            description=(
                "Recommends crops to sow based on demand signals, seasonality, and price forecasts"
            ),
            temperature=0.5,
            max_tokens=600,
        )
        super().__init__(config=config, llm=llm, **kwargs)
        self._engine = None

    async def initialize(self) -> bool:
        """Create the shared ADCL service."""
        try:
            from src.agents.adcl import get_adcl_service

            self._engine = get_adcl_service(llm=self.llm)
            self._initialized = True
            logger.info("ADCLWrapperAgent initialized")
        except Exception as exc:
            logger.warning("ADCL engine init failed: {}", exc)
            self._initialized = True
        return True

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        return build_system_prompt(
            role_description="You are CropFresh's Crop Recommendation Agent.",
            domain_prompt=(
                "Help farmers decide what to sow based on market demand, "
                "seasonality, and price forecasts for Karnataka. "
                "Use the ADCL engine to generate data-backed recommendations."
            ),
            context=context,
            agent_domain="adcl",
        )

    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution: Optional[AgentExecutionState] = None,
    ) -> AgentResponse:
        """Generate crop recommendations or fall back to the LLM."""
        del execution
        district = self._extract_district(query, context)
        if self._engine:
            try:
                report = await self._engine.generate_weekly_report(district=district)
                return AgentResponse(
                    content=format_weekly_report(report, district),
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

        if self.llm:
            messages = [
                {"role": "system", "content": self._get_system_prompt(context)},
                {"role": "user", "content": query},
            ]
            answer = await self.generate_with_llm(messages, context=context)
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

    def _extract_district(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> str:
        """Extract district from query text or user profile."""
        if context:
            profile = context.get("user_profile", {})
            if profile.get("district"):
                return profile["district"]

        query_lower = query.lower()
        for district in DISTRICTS:
            if district in query_lower:
                return district.title()
        return "Bangalore"
