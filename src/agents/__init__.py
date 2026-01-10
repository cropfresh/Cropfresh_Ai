"""
Agents Module
=============
Multi-agent system for CropFresh AI.

Provides:
- SupervisorAgent: Central orchestrator
- AgronomyAgent: Farming expertise
- CommerceAgent: Market intelligence
- PlatformAgent: App support
- GeneralAgent: Fallback handler

Author: CropFresh AI Team
Version: 2.0.0
"""

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.supervisor_agent import SupervisorAgent, RoutingDecision
from src.agents.agronomy_agent import AgronomyAgent
from src.agents.commerce_agent import CommerceAgent
from src.agents.platform_agent import PlatformAgent
from src.agents.general_agent import GeneralAgent

# Legacy agents
from src.agents.knowledge_agent import KnowledgeAgent, KnowledgeResponse
from src.agents.pricing_agent import PricingAgent, PriceRecommendation

__all__ = [
    # Base
    "BaseAgent",
    "AgentConfig",
    "AgentResponse",
    
    # Multi-Agent System
    "SupervisorAgent",
    "RoutingDecision",
    "AgronomyAgent",
    "CommerceAgent",
    "PlatformAgent",
    "GeneralAgent",
    
    # Legacy
    "KnowledgeAgent",
    "KnowledgeResponse",
    "PricingAgent",
    "PriceRecommendation",
]
