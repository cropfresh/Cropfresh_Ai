"""Supervisor dependency builder for the shared chat API."""

from __future__ import annotations

from src.config import get_settings

_supervisor_agent = None


async def get_supervisor_agent():
    """Get or create the shared supervisor agent instance."""
    global _supervisor_agent

    if _supervisor_agent is not None:
        return _supervisor_agent

    from loguru import logger

    from src.agents.agronomy_agent import AgronomyAgent
    from src.agents.buyer_matching.agent import BuyerMatchingAgent
    from src.agents.commerce_agent import CommerceAgent
    from src.agents.general_agent import GeneralAgent
    from src.agents.platform_agent import PlatformAgent
    from src.agents.quality_assessment.agent import QualityAssessmentAgent
    from src.agents.supervisor import SupervisorAgent
    from src.memory.state_manager import AgentStateManager
    from src.orchestrator.llm_provider import create_llm_provider
    from src.rag.knowledge_base import KnowledgeBase
    from src.tools.registry import get_tool_registry

    settings = get_settings()

    llm = None
    if settings.has_llm_configured:
        llm = create_llm_provider(
            provider=settings.llm_provider,
            api_key=settings.groq_api_key or settings.together_api_key,
            base_url=getattr(settings, "vllm_base_url", ""),
            model=settings.llm_model,
            region=getattr(settings, "aws_region", "ap-south-1"),
            aws_profile=getattr(settings, "aws_profile", ""),
        )

    kb = None
    try:
        kb = KnowledgeBase(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
        if not await kb.initialize():
            logger.warning("KnowledgeBase init returned False; chat continues without RAG")
            kb = None
    except Exception as exc:
        logger.warning("KnowledgeBase init failed ({}); chat continues without RAG", exc)
        kb = None

    state_manager = AgentStateManager()
    tool_registry = get_tool_registry()
    _supervisor_agent = SupervisorAgent(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=kb,
    )

    agronomy = AgronomyAgent(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=kb,
    )
    commerce = CommerceAgent(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=kb,
    )
    buyer_matching = BuyerMatchingAgent(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=kb,
        redis_url=settings.redis_url,
    )
    quality_assessment = QualityAssessmentAgent(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=kb,
    )
    platform = PlatformAgent(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=kb,
    )
    general = GeneralAgent(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=kb,
    )

    _supervisor_agent.register_agent("agronomy_agent", agronomy)
    _supervisor_agent.register_agent("commerce_agent", commerce)
    _supervisor_agent.register_agent("buyer_matching_agent", buyer_matching)
    _supervisor_agent.register_agent("quality_assessment_agent", quality_assessment)
    _supervisor_agent.register_agent("platform_agent", platform)
    _supervisor_agent.register_agent("general_agent", general)
    _supervisor_agent.set_fallback_agent(general)
    await _supervisor_agent.initialize()
    return _supervisor_agent
