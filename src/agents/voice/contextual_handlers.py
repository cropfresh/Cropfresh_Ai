"""Contextual voice handlers extracted from handlers_ext.py."""

from __future__ import annotations

from loguru import logger


async def handle_check_weather(agent, template, entities, session):
    """Handle check_weather via the configured weather tool."""
    location = entities.get("location", session.context.get("location", "Kolar"))
    if agent.weather_tool:
        try:
            forecast = await agent.weather_tool.get_forecast(location=location)
            if hasattr(forecast, "current"):
                current = forecast.current
                return template.format(
                    location=location,
                    condition=getattr(current, "condition", "Clear"),
                    temp=getattr(current, "temperature_c", 28),
                    advisory=getattr(forecast, "planting_advisory", ""),
                )
            return template.format(
                location=location,
                condition=forecast.get("condition", "Clear"),
                temp=forecast.get("temperature", 28),
                advisory=forecast.get("advisory", ""),
            )
        except Exception as exc:
            logger.warning("Voice weather lookup failed: {}", exc)

    if session.language == "hi":
        return f"{location} ke liye mausam seva abhi uplabdh nahin hai."
    if session.language == "kn":
        return f"{location} gaagi havaamana seve labhyavilla."
    return f"Weather service is not available right now for {location}."


async def handle_get_advisory(agent, template, entities, session):
    """Handle get_advisory by querying the agronomy agent."""
    crop = entities.get("crop", "")
    if not crop:
        if session.language == "hi":
            return "Kis fasal ke baare mein salah chahiye?"
        if session.language == "kn":
            return "Yaava beleya bagge salahhe beku?"
        return "Which crop do you need advice for?"

    if agent.agronomy_agent:
        try:
            response = await agent.agronomy_agent.process(
                f"Give brief farming advice for {crop}",
                context={"language": session.language},
            )
            advisory_text = getattr(response, "content", str(response))
            return template.format(crop=crop, advisory=advisory_text[:200])
        except Exception as exc:
            logger.warning("Voice advisory lookup failed: {}", exc)

    return template.format(crop=crop, advisory="No advisory available at this time.")


async def handle_weekly_demand(agent, template, entities, session):
    """Handle weekly_demand using the canonical ADCL report when possible."""
    location = entities.get("location", session.context.get("location", "Karnataka"))
    if agent.adcl_agent:
        try:
            if hasattr(agent.adcl_agent, "get_weekly_demand"):
                report = await agent.adcl_agent.get_weekly_demand(district=location)
                crops = report.get("crops", [])
                demand_list = ", ".join(crop.get("commodity", "") for crop in crops[:5] if crop.get("commodity"))
                if demand_list:
                    return template.format(location=location, demand_list=demand_list)
            if hasattr(agent.adcl_agent, "get_weekly_list"):
                demand = await agent.adcl_agent.get_weekly_list(location=location)
                demand_list = demand if isinstance(demand, str) else ", ".join(str(item) for item in demand)
                return template.format(location=location, demand_list=demand_list)
        except Exception as exc:
            logger.warning("Voice weekly demand lookup failed: {}", exc)

    if session.language == "hi":
        return f"{location} ke liye saptaahik maang seva abhi uplabdh nahin hai."
    if session.language == "kn":
        return f"{location} saaptaahika bedike seve labhyavilla."
    return f"Weekly demand service is not available right now for {location}."
