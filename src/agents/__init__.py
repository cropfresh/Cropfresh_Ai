"""Agents package public exports with lazy loading to avoid import side effects."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ADCLWrapperAgent": "src.agents.adcl_wrapper_agent.ADCLWrapperAgent",
    "AgentConfig": "src.agents.base_agent.AgentConfig",
    "AgentResponse": "src.agents.base_agent.AgentResponse",
    "AgronomyAgent": "src.agents.agronomy_agent.AgronomyAgent",
    "BaseAgent": "src.agents.base_agent.BaseAgent",
    "CommerceAgent": "src.agents.commerce_agent.CommerceAgent",
    "GeneralAgent": "src.agents.general_agent.GeneralAgent",
    "KnowledgeAgent": "src.agents.knowledge_agent.KnowledgeAgent",
    "KnowledgeResponse": "src.agents.knowledge_agent.KnowledgeResponse",
    "LogisticsWrapperAgent": "src.agents.logistics_wrapper_agent.LogisticsWrapperAgent",
    "PlatformAgent": "src.agents.platform_agent.PlatformAgent",
    "PriceRecommendation": "src.agents.pricing_agent.PriceRecommendation",
    "PricingAgent": "src.agents.pricing_agent.PricingAgent",
    "RoutingDecision": "src.agents.supervisor.RoutingDecision",
    "SupervisorAgent": "src.agents.supervisor.SupervisorAgent",
    "create_agent_system": "src.agents.agent_registry.create_agent_system",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.agents' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name].rsplit(".", 1)
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
