"""Assembly helpers for the shared agent tool registry."""

from __future__ import annotations

from loguru import logger

from src.agents.tool_setup.agronomy import register_agronomy_tools
from src.agents.tool_setup.commerce import register_commerce_tools
from src.agents.tool_setup.rates import register_rate_tools
from src.agents.tool_setup.research import register_research_tools
from src.tools.registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    """Create and populate the shared tool registry."""
    registry = ToolRegistry()
    for register in (
        register_commerce_tools,
        register_agronomy_tools,
        register_research_tools,
        register_rate_tools,
    ):
        register(registry)

    logger.info("ToolRegistry created with {} tools", len(registry.list_tools()))
    return registry
