"""Agronomy tool registration for agent-facing tool registries."""

from __future__ import annotations

from loguru import logger

from src.tools.imd_weather import get_imd_client
from src.tools.registry import ToolRegistry
from src.tools.weather import WeatherTool


async def _imd_weather(
    state: str = "Karnataka",
    district: str = "Bengaluru",
    days: int = 5,
    include_advisory: bool = True,
) -> dict:
    client = get_imd_client(use_mock=True)
    forecast = await client.get_forecast(state=state, district=district, days=days)
    payload = forecast.model_dump(mode="json")
    if include_advisory:
        payload["advisory"] = (
            await client.get_agro_advisory(state=state, district=district)
        ).model_dump(mode="json")
    return payload


async def _get_weather(location: str, days: int = 1) -> dict:
    tool = WeatherTool()
    if days <= 1:
        return (await tool.get_current(location)).model_dump(mode="json")
    return (await tool.get_forecast(location, days)).model_dump(mode="json")


def register_agronomy_tools(registry: ToolRegistry) -> None:
    """Register weather and advisory tools."""
    try:
        registry.add_tool(
            _imd_weather,
            name="imd_weather",
            description="Get district-level IMD-style forecast and agro advisory data.",
            category="agronomy",
        )
        registry.add_tool(
            _get_weather,
            name="get_weather",
            description="Get current weather or short forecast for a location.",
            category="agronomy",
        )
    except Exception as exc:
        logger.debug("Agronomy tool registration skipped: {}", exc)
