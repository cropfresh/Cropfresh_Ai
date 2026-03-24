"""Rate-hub tool registration for agent-facing tool registries."""

from __future__ import annotations

from loguru import logger

from src.tools.multi_source_rates import multi_source_rates_tool, price_api_tool
from src.tools.registry import ToolRegistry


def register_rate_tools(registry: ToolRegistry) -> None:
    """Register multi-source rate tools."""
    try:
        registry.add_tool(
            multi_source_rates_tool,
            name="multi_source_rates",
            description="Fetch Karnataka mandi, fuel, gold, or support-price data from multiple sources.",
            category="rates",
        )
        registry.add_tool(
            price_api_tool,
            name="price_api",
            description="Backward-compatible mandi-only alias for multi-source rate queries.",
            category="rates",
        )
    except Exception as exc:
        logger.debug("Rate tool registration skipped: {}", exc)
