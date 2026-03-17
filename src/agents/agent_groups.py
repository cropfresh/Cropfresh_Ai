"""Helpers for building grouped agent instances."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from loguru import logger


def build_shared_agent_kwargs(
    llm: Any,
    tool_registry: Any,
    state_manager: Any,
    knowledge_base: Any,
    settings: Any = None,
) -> dict[str, Any]:
    """Build the shared kwargs dictionary passed across agent groups."""
    return {
        "llm": llm,
        "tool_registry": tool_registry,
        "state_manager": state_manager,
        "knowledge_base": knowledge_base,
        "settings": settings,
    }


def create_all_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Instantiate every supported agent group and merge the results."""
    all_agents: dict[str, Any] = {}
    for builder in (
        _create_core_agents,
        _create_pricing_agents,
        _create_marketplace_agents,
        _create_web_agents,
        _create_wrapper_agents,
        _create_knowledge_agent_adapter,
    ):
        all_agents.update(builder(kwargs))
    return all_agents


def _load_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def _add_agent(
    agents: dict[str, Any],
    name: str,
    label: str,
    factory: Callable[[], Any],
) -> None:
    try:
        agents[name] = factory()
    except Exception as exc:
        logger.warning("{} creation failed: {}", label, exc)


def _create_core_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    agent_kwargs = {key: value for key, value in kwargs.items() if key != "settings"}
    specs = [
        ("agronomy_agent", "AgronomyAgent", "src.agents.agronomy_agent.AgronomyAgent"),
        ("commerce_agent", "CommerceAgent", "src.agents.commerce_agent.CommerceAgent"),
        ("platform_agent", "PlatformAgent", "src.agents.platform_agent.PlatformAgent"),
        ("general_agent", "GeneralAgent", "src.agents.general_agent.GeneralAgent"),
    ]
    for name, label, class_path in specs:
        _add_agent(
            agents,
            name,
            label,
            lambda class_path=class_path: _load_class(class_path)(**agent_kwargs),
        )
    return agents


def _create_pricing_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    agents: dict[str, Any] = {}

    def pricing_agent():
        agent = _load_class("src.agents.pricing_agent.PricingAgent")(llm=kwargs["llm"])
        agent.state_manager = kwargs["state_manager"]
        return agent

    def price_prediction_agent():
        return _load_class("src.agents.price_prediction.agent.PricePredictionAgent")(
            llm=kwargs["llm"],
            state_manager=kwargs["state_manager"],
        )

    _add_agent(agents, "pricing_agent", "PricingAgent", pricing_agent)
    _add_agent(
        agents,
        "price_prediction_agent",
        "PricePredictionAgent",
        price_prediction_agent,
    )
    return agents


def _create_marketplace_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    specs = [
        (
            "buyer_matching_agent",
            "BuyerMatchingAgent",
            "src.agents.buyer_matching.agent.BuyerMatchingAgent",
        ),
        (
            "quality_assessment_agent",
            "QualityAssessmentAgent",
            "src.agents.quality_assessment.agent.QualityAssessmentAgent",
        ),
        (
            "crop_listing_agent",
            "CropListingAgent",
            "src.agents.crop_listing.agent.CropListingAgent",
        ),
    ]
    for name, label, class_path in specs:
        _add_agent(
            agents,
            name,
            label,
            lambda class_path=class_path: _load_class(class_path)(
                llm=kwargs["llm"],
                state_manager=kwargs["state_manager"],
            ),
        )
    return agents


def _create_web_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    agents: dict[str, Any] = {}

    def web_scraping_agent():
        agent = _load_class("src.agents.web_scraping_agent.WebScrapingAgent")(
            llm_provider=kwargs["llm"]
        )
        agent.state_manager = kwargs["state_manager"]
        return agent

    def browser_agent():
        agent = _load_class("src.agents.browser_agent.BrowserAgent")()
        agent.state_manager = kwargs["state_manager"]
        return agent

    def research_agent():
        return _load_class("src.agents.research.research_agent.ResearchAgent")(
            llm=kwargs["llm"],
            knowledge_base=kwargs["knowledge_base"],
            state_manager=kwargs["state_manager"],
        )

    _add_agent(agents, "web_scraping_agent", "WebScrapingAgent", web_scraping_agent)
    _add_agent(agents, "browser_agent", "BrowserAgent", browser_agent)
    _add_agent(agents, "research_agent", "ResearchAgent", research_agent)
    return agents


def _create_wrapper_agents(kwargs: dict[str, Any]) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    _add_agent(
        agents,
        "adcl_agent",
        "ADCLWrapperAgent",
        lambda: _load_class("src.agents.adcl_wrapper_agent.ADCLWrapperAgent")(
            llm=kwargs["llm"],
            state_manager=kwargs["state_manager"],
        ),
    )
    _add_agent(
        agents,
        "logistics_agent",
        "LogisticsWrapperAgent",
        lambda: _load_class("src.agents.logistics_wrapper_agent.LogisticsWrapperAgent")(
            state_manager=kwargs["state_manager"]
        ),
    )
    return agents


def _create_knowledge_agent_adapter(kwargs: dict[str, Any]) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    settings = kwargs.get("settings")

    def knowledge_agent():
        agent = _load_class("src.agents.knowledge_agent.KnowledgeAgent")(
            llm=kwargs["llm"],
            qdrant_host=getattr(settings, "qdrant_host", "") or "localhost",
            qdrant_port=getattr(settings, "qdrant_port", 6333) or 6333,
            qdrant_api_key=getattr(settings, "qdrant_api_key", "") or "",
        )
        agent.state_manager = kwargs["state_manager"]
        return agent

    _add_agent(agents, "knowledge_agent", "KnowledgeAgent", knowledge_agent)
    return agents
