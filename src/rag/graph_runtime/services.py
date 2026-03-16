from __future__ import annotations

from typing import Any

from src.rag.routing.prefilter import extract_entities


def _first_value(values: list[str], default: str = "") -> str:
    return values[0] if values else default


async def retrieve_vector_documents(
    knowledge_base: Any,
    queries: list[str],
    top_k: int = 5,
) -> list[Any]:
    """Retrieve and de-duplicate vector search documents."""
    if knowledge_base is None:
        return []

    all_docs: list[Any] = []
    seen_ids: set[str] = set()
    for query in queries[:3]:
        result = await knowledge_base.search(query, top_k=top_k)
        for doc in result.documents:
            if doc.id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc.id)
    return all_docs


async def retrieve_live_price_documents(query: str) -> list[Any]:
    """Retrieve live mandi prices for price-centric queries."""
    from src.rag.knowledge_base import Document
    from src.tools.agmarknet import AgmarknetTool

    entities = extract_entities(query)
    commodity = _first_value(entities["crops"], "Tomato").title()
    district = _first_value(entities["locations"], "Kolar")
    tool = AgmarknetTool()
    prices = await tool.get_prices(commodity=commodity, state="Karnataka", district=district)
    if not prices:
        prices = tool.get_mock_prices(commodity=commodity, district=district)
    return [
        Document(
            id=f"price_{price.market}_{idx}",
            text=(
                f"{price.commodity} mandi price in {price.market}, {price.district}: "
                f"min Rs.{price.min_price:.0f}, max Rs.{price.max_price:.0f}, "
                f"modal Rs.{price.modal_price:.0f} per quintal as of {price.date.date()}."
            ),
            source="agmarknet",
            metadata={
                "source": "agmarknet",
                "market": price.market,
                "district": price.district,
                "commodity": price.commodity,
                "timestamp": price.date.timestamp(),
                "as_of": price.date.isoformat(),
            },
            score=1.0,
        )
        for idx, price in enumerate(prices)
    ]


async def retrieve_weather_documents(query: str) -> list[Any]:
    """Retrieve weather forecast context for weather queries."""
    from src.rag.knowledge_base import Document
    from src.tools.weather import WeatherTool

    location = _first_value(extract_entities(query)["locations"], "Kolar")
    forecast = await WeatherTool(use_mock=True).get_forecast(location=location, days=3)
    return [
        Document(
            id=f"weather_{location.lower()}",
            text=(
                f"Weather forecast for {forecast.location}: "
                f"current {forecast.current.temperature_c}C and "
                f"{forecast.current.rainfall_mm}mm rainfall."
            ),
            source="weather",
            metadata={"source": "weather", "timestamp": forecast.current.date.timestamp()},
            score=1.0,
        )
    ]


async def retrieve_browser_documents(query: str, web_search_tool: Any) -> list[Any]:
    """Retrieve live web documents through the configured browser/search tool."""
    if web_search_tool is None:
        return []

    from src.rag.knowledge_base import Document

    results = await web_search_tool.search(query)
    return [
        Document(
            id=f"browser_{index}",
            text=str(result),
            source="browser_search",
            metadata={"source": "browser_search"},
            score=0.6,
        )
        for index, result in enumerate(results)
    ]


async def generate_answer(query: str, documents: list[Any], llm: Any) -> tuple[str, str]:
    """Generate an answer, falling back to grounded extractive text without an LLM."""
    if not documents:
        return "I don't have enough information to answer.", "none"

    if llm is None:
        snippets = " ".join(getattr(doc, "text", str(doc)) for doc in documents[:2])
        return snippets[:500], "extractive_fallback"

    from src.rag.agentic.speculative import SpeculativeDraftEngine

    engine = SpeculativeDraftEngine(drafter_llm=llm, verifier_llm=llm)
    answer, _ = await engine.generate_and_select(documents=documents, query=query)
    return answer, "speculative_3x"
