"""Central factory that wires the supervisor and domain agents."""

from typing import Any, Optional

from loguru import logger

from src.agents.agent_groups import build_shared_agent_kwargs, create_all_agents
from src.agents.supervisor import SupervisorAgent
from src.agents.tool_registry_setup import build_tool_registry
from src.memory.state_manager import AgentStateManager


async def create_agent_system(
    llm: Any = None,
    knowledge_base: Any = None,
    redis_url: Optional[str] = None,
    settings: Any = None,
) -> tuple[SupervisorAgent, AgentStateManager]:
    """Create the complete agent system with all agents registered."""
    state_manager = _create_state_manager(redis_url)
    tool_registry = build_tool_registry()
    kwargs = build_shared_agent_kwargs(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=knowledge_base,
        settings=settings,
    )
    all_agents = create_all_agents(kwargs)

    supervisor = SupervisorAgent(
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager,
        knowledge_base=knowledge_base,
    )
    for name, agent in all_agents.items():
        supervisor.register_agent(name, agent)

    if "general_agent" in all_agents:
        supervisor.set_fallback_agent(all_agents["general_agent"])

    await supervisor.initialize()
    logger.info(
        "Agent system ready with {} agents: {}",
        len(all_agents),
        ", ".join(sorted(all_agents.keys())),
    )
    return supervisor, state_manager


def _create_state_manager(redis_url: Optional[str] = None) -> AgentStateManager:
    """Create the shared state manager for session and memory tracking."""
    manager = AgentStateManager(redis_url=redis_url)
    logger.info("AgentStateManager created (redis={})", bool(redis_url))
    return manager
