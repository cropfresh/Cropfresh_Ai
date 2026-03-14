"""
Agents Module
=============
Multi-agent system for CropFresh AI.

Provides:
- SupervisorAgent: Central orchestrator
- create_agent_system: Factory that wires ALL agents at startup
- Domain agents: Agronomy, Commerce, Platform, General
- Wrapper agents: ADCL, Logistics (bridge standalone engines)

Author: CropFresh AI Team
Version: 2.1.0
"""

# * Wrapper agents for standalone engines
from src.agents.adcl_wrapper_agent import ADCLWrapperAgent
from src.agents.agent_registry import create_agent_system
from src.agents.agronomy_agent import AgronomyAgent
from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.commerce_agent import CommerceAgent
from src.agents.general_agent import GeneralAgent

# Legacy agents
from src.agents.knowledge_agent import KnowledgeAgent, KnowledgeResponse
from src.agents.logistics_wrapper_agent import LogisticsWrapperAgent
from src.agents.platform_agent import PlatformAgent
from src.agents.pricing_agent import PriceRecommendation, PricingAgent
from src.agents.supervisor import RoutingDecision, SupervisorAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentConfig",
    "AgentResponse",

    # System factory
    "create_agent_system",

    # Multi-Agent System
    "SupervisorAgent",
    "RoutingDecision",
    "AgronomyAgent",
    "CommerceAgent",
    "PlatformAgent",
    "GeneralAgent",

    # Wrapper agents
    "ADCLWrapperAgent",
    "LogisticsWrapperAgent",

    # Legacy
    "KnowledgeAgent",
    "KnowledgeResponse",
    "PricingAgent",
    "PriceRecommendation",
]

