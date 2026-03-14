"""
Agent Registry
==============
Central factory that instantiates and wires ALL agents
into the SupervisorAgent at startup.

Resolves the critical gap where main.py created a bare
SupervisorAgent with zero registered agents.

Author: CropFresh AI Team
Version: 2.1.0
"""

from typing import Any, Optional

from loguru import logger

from src.agents.supervisor import SupervisorAgent
from src.memory.state_manager import AgentStateManager
from src.tools.registry import ToolRegistry

# * ═══════════════════════════════════════════════════════════════
# * SHARED DEPENDENCIES — created once, injected into all agents
# * ═══════════════════════════════════════════════════════════════

def _create_state_manager(redis_url: Optional[str] = None) -> AgentStateManager:
    """Create the shared state manager for session/memory tracking."""
    manager = AgentStateManager(redis_url=redis_url)
    logger.info("AgentStateManager created (redis={})", bool(redis_url))
    return manager


def _create_tool_registry() -> ToolRegistry:
    """Create and populate the shared tool registry."""
    registry = ToolRegistry()

    # * Register tools that are available — each import guarded
    _register_commerce_tools(registry)
    _register_agronomy_tools(registry)
    _register_research_tools(registry)

    logger.info("ToolRegistry created with {} tools", len(registry.list_tools()))
    return registry


# * ═══════════════════════════════════════════════════════════════
# * TOOL REGISTRATION — grouped by domain
# * ═══════════════════════════════════════════════════════════════

def _register_commerce_tools(registry: ToolRegistry) -> None:
    """Register commerce/market tools (Agmarknet, ML forecaster, etc.)."""
    try:
        from src.tools.agmarknet import AgmarknetTool
        registry.register("agmarknet", AgmarknetTool(), category="commerce")
    except Exception as exc:
        logger.debug("Agmarknet tool skipped: {}", exc)

    try:
        from src.tools.ml_forecaster import PriceForecaster
        registry.register("ml_forecaster", PriceForecaster(), category="commerce")
    except Exception as exc:
        logger.debug("ML forecaster tool skipped: {}", exc)

    try:
        from src.tools.news_sentiment import NewsSentimentScraper
        registry.register("news_sentiment", NewsSentimentScraper(), category="commerce")
    except Exception as exc:
        logger.debug("News sentiment tool skipped: {}", exc)


def _register_agronomy_tools(registry: ToolRegistry) -> None:
    """Register agronomy tools (weather, etc.)."""
    try:
        from src.tools.imd_weather import IMDWeatherClient
        registry.register("imd_weather", IMDWeatherClient(), category="agronomy")
    except Exception as exc:
        logger.debug("IMD weather tool skipped: {}", exc)

    try:
        from src.tools.weather import WeatherTool
        registry.register("get_weather", WeatherTool(), category="agronomy")
    except Exception as exc:
        logger.debug("Weather tool skipped: {}", exc)


def _register_research_tools(registry: ToolRegistry) -> None:
    """Register research/general tools."""
    try:
        from src.tools.deep_research import DeepResearchTool
        registry.register("deep_research", DeepResearchTool(), category="research")
    except Exception as exc:
        logger.debug("Deep research tool skipped: {}", exc)

    try:
        from src.tools.web_search import WebSearchTool
        registry.register("web_search", WebSearchTool(), category="general")
    except Exception as exc:
        logger.debug("Web search tool skipped: {}", exc)


# * ═══════════════════════════════════════════════════════════════
# * AGENT INSTANTIATION — one function per group for modularity
# * ═══════════════════════════════════════════════════════════════

def _shared_kwargs(
    llm: Any,
    tool_registry: ToolRegistry,
    state_manager: AgentStateManager,
    knowledge_base: Any,
    settings: Any = None,
) -> dict[str, Any]:
    """Build the common kwargs dict passed to every BaseAgent."""
    return {
        "llm": llm,
        "tool_registry": tool_registry,
        "state_manager": state_manager,
        "knowledge_base": knowledge_base,
        "settings": settings,
    }


def _create_core_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Instantiate core domain agents (Agronomy, Commerce, Platform, General).

    Returns dict of agent_name → agent_instance.
    """
    agents: dict[str, Any] = {}
    # BaseAgent.__init__ doesn't accept 'settings', so exclude it from **kwargs
    agent_kwargs = {k: v for k, v in kwargs.items() if k != "settings"}

    try:
        from src.agents.agronomy_agent import AgronomyAgent
        agents["agronomy_agent"] = AgronomyAgent(**agent_kwargs)
    except Exception as exc:
        logger.warning("AgronomyAgent creation failed: {}", exc)

    try:
        from src.agents.commerce_agent import CommerceAgent
        agents["commerce_agent"] = CommerceAgent(**agent_kwargs)
    except Exception as exc:
        logger.warning("CommerceAgent creation failed: {}", exc)

    try:
        from src.agents.platform_agent import PlatformAgent
        agents["platform_agent"] = PlatformAgent(**agent_kwargs)
    except Exception as exc:
        logger.warning("PlatformAgent creation failed: {}", exc)

    try:
        from src.agents.general_agent import GeneralAgent
        agents["general_agent"] = GeneralAgent(**agent_kwargs)
    except Exception as exc:
        logger.warning("GeneralAgent creation failed: {}", exc)

    return agents


def _create_pricing_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Instantiate pricing-related agents (Pricing DPLE, PricePrediction)."""
    agents: dict[str, Any] = {}

    try:
        from src.agents.pricing_agent import PricingAgent
        agent = PricingAgent(llm=kwargs["llm"])
        # * PricingAgent doesn't inherit BaseAgent — attach state_manager as attribute
        agent.state_manager = kwargs["state_manager"]
        agents["pricing_agent"] = agent
    except Exception as exc:
        logger.warning("PricingAgent creation failed: {}", exc)

    try:
        from src.agents.price_prediction.agent import PricePredictionAgent
        # * PricePredictionAgent uses **kwargs → state_manager flows through
        agents["price_prediction_agent"] = PricePredictionAgent(
            llm=kwargs["llm"],
            state_manager=kwargs["state_manager"],
        )
    except Exception as exc:
        logger.warning("PricePredictionAgent creation failed: {}", exc)

    return agents


def _create_marketplace_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Instantiate marketplace agents (BuyerMatching, CropListing, Quality)."""
    agents: dict[str, Any] = {}

    try:
        from src.agents.buyer_matching.agent import BuyerMatchingAgent
        # * BuyerMatchingAgent uses **kwargs → state_manager flows through
        agents["buyer_matching_agent"] = BuyerMatchingAgent(
            llm=kwargs["llm"],
            state_manager=kwargs["state_manager"],
        )
    except Exception as exc:
        logger.warning("BuyerMatchingAgent creation failed: {}", exc)

    try:
        from src.agents.quality_assessment.agent import QualityAssessmentAgent
        agents["quality_assessment_agent"] = QualityAssessmentAgent(
            llm=kwargs["llm"],
            state_manager=kwargs["state_manager"],
        )
    except Exception as exc:
        logger.warning("QualityAssessmentAgent creation failed: {}", exc)

    try:
        from src.agents.crop_listing.agent import CropListingAgent
        # * CropListingAgent uses **kwargs → state_manager flows through
        agents["crop_listing_agent"] = CropListingAgent(
            llm=kwargs["llm"],
            state_manager=kwargs["state_manager"],
        )
    except Exception as exc:
        logger.warning("CropListingAgent creation failed: {}", exc)

    return agents


def _create_web_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Instantiate web/research agents (Scraping, Browser, Research)."""
    agents: dict[str, Any] = {}

    try:
        from src.agents.web_scraping_agent import WebScrapingAgent
        # * WebScrapingAgent doesn't inherit BaseAgent — attach state_manager as attribute
        agent = WebScrapingAgent(llm_provider=kwargs["llm"])
        agent.state_manager = kwargs["state_manager"]
        agents["web_scraping_agent"] = agent
    except Exception as exc:
        logger.warning("WebScrapingAgent creation failed: {}", exc)

    try:
        from src.agents.browser_agent import BrowserAgent
        # * BrowserAgent doesn't inherit BaseAgent — attach state_manager as attribute
        agent = BrowserAgent()
        agent.state_manager = kwargs["state_manager"]
        agents["browser_agent"] = agent
    except Exception as exc:
        logger.warning("BrowserAgent creation failed: {}", exc)

    try:
        from src.agents.research.research_agent import ResearchAgent
        agents["research_agent"] = ResearchAgent(
            llm=kwargs["llm"],
            knowledge_base=kwargs["knowledge_base"],
            state_manager=kwargs["state_manager"],
        )
    except Exception as exc:
        logger.warning("ResearchAgent creation failed: {}", exc)

    return agents


def _create_wrapper_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Instantiate wrapper agents for standalone engines.

    ADCL and LogisticsRouter do NOT inherit BaseAgent,
    so we wrap them in lightweight adapter agents.
    Both use **kwargs → state_manager flows through automatically.
    """
    agents: dict[str, Any] = {}

    try:
        from src.agents.adcl_wrapper_agent import ADCLWrapperAgent
        agents["adcl_agent"] = ADCLWrapperAgent(
            llm=kwargs["llm"],
            state_manager=kwargs["state_manager"],
        )
    except Exception as exc:
        logger.warning("ADCLWrapperAgent creation failed: {}", exc)

    try:
        from src.agents.logistics_wrapper_agent import LogisticsWrapperAgent
        agents["logistics_agent"] = LogisticsWrapperAgent(
            state_manager=kwargs["state_manager"],
        )
    except Exception as exc:
        logger.warning("LogisticsWrapperAgent creation failed: {}", exc)

    return agents


def _create_knowledge_agent_adapter(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Register KnowledgeAgent so the supervisor can route
    'deep knowledge' queries to the full RAG pipeline.
    KnowledgeAgent doesn't inherit BaseAgent — attach state_manager as attribute.
    """
    agents: dict[str, Any] = {}

    try:
        from src.agents.knowledge_agent import KnowledgeAgent
        # Pull Qdrant Cloud config from settings (avoid defaulting to localhost)
        settings = kwargs.get("settings")
        qdrant_host = getattr(settings, "qdrant_host", "") or "localhost"
        qdrant_port = getattr(settings, "qdrant_port", 6333) or 6333
        qdrant_api_key = getattr(settings, "qdrant_api_key", "") or ""
        agent = KnowledgeAgent(
            llm=kwargs["llm"],
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_api_key=qdrant_api_key,
        )
        # * KnowledgeAgent doesn't inherit BaseAgent — attach state_manager as attribute
        agent.state_manager = kwargs["state_manager"]
        agents["knowledge_agent"] = agent
    except Exception as exc:
        logger.warning("KnowledgeAgent adapter creation failed: {}", exc)

    return agents


# * ═══════════════════════════════════════════════════════════════
# * PUBLIC API — called from main.py lifespan
# * ═══════════════════════════════════════════════════════════════

async def create_agent_system(
    llm: Any = None,
    knowledge_base: Any = None,
    redis_url: Optional[str] = None,
    settings: Any = None,
) -> tuple[SupervisorAgent, AgentStateManager]:
    """
    Create the complete agent system with all agents registered.

    This is the single entry point for main.py lifespan. It:
      1. Creates shared infrastructure (StateManager, ToolRegistry)
      2. Instantiates all domain agents with shared deps
      3. Registers each with SupervisorAgent
      4. Returns fully-wired supervisor + state_manager

    Args:
        llm:            Shared LLM provider instance.
        knowledge_base: Shared KnowledgeBase (Qdrant) instance.
        redis_url:      Redis URL for session persistence.
        settings:       Application settings (for future config).

    Returns:
        Tuple of (SupervisorAgent, AgentStateManager).
    """
    # * 1. Shared infrastructure
    state_manager = _create_state_manager(redis_url)
    tool_registry = _create_tool_registry()

    kwargs = _shared_kwargs(llm, tool_registry, state_manager, knowledge_base, settings)

    # * 2. Instantiate all agent groups
    all_agents: dict[str, Any] = {}
    all_agents.update(_create_core_agents(kwargs))
    all_agents.update(_create_pricing_agents(kwargs))
    all_agents.update(_create_marketplace_agents(kwargs))
    all_agents.update(_create_web_agents(kwargs))
    all_agents.update(_create_wrapper_agents(kwargs))
    all_agents.update(_create_knowledge_agent_adapter(kwargs))

    # * 3. Create supervisor with shared deps
    supervisor = SupervisorAgent(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=knowledge_base,
    )

    # * 4. Register each agent with supervisor
    for name, agent in all_agents.items():
        supervisor.register_agent(name, agent)

    # * 5. Set general_agent as fallback
    if "general_agent" in all_agents:
        supervisor.set_fallback_agent(all_agents["general_agent"])

    # * 6. Initialize supervisor (cascades to all registered agents)
    await supervisor.initialize()

    logger.info(
        "✅ Agent system ready — {} agents registered: {}",
        len(all_agents),
        ", ".join(sorted(all_agents.keys())),
    )

    return supervisor, state_manager
