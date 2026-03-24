"""Tool handlers for the standalone agentic orchestrator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from loguru import logger

from src.rates import ComparisonDepth, RateKind
from src.rates.factory import get_rate_service
from src.rates.query_builder import normalize_rate_query
from src.rates.settings import get_agmarknet_api_key


class AgenticToolHandlers:
    """Small, testable tool handler bundle used by AgenticOrchestrator."""

    def __init__(self, knowledge_base=None, graph_retriever=None, weather_client=None, browser_rag=None):
        self.knowledge_base = knowledge_base
        self.graph_retriever = graph_retriever
        self.weather_client = weather_client
        self.browser_rag = browser_rag

    async def vector_search(self, params: dict) -> list[Any]:
        if self.knowledge_base is None:
            return []
        try:
            result = await self.knowledge_base.search(params.get("query", ""), top_k=params.get("top_k", 5))
            return result.documents if hasattr(result, "documents") else []
        except Exception as exc:
            logger.warning("vector_search tool failed: {}", exc)
            return []

    async def graph_rag(self, params: dict) -> list[Any]:
        if self.graph_retriever is None:
            return []
        try:
            query = params.get("query", params.get("entity", ""))
            graph_ctx = await self.graph_retriever.retrieve(query)
            if graph_ctx and graph_ctx.context_text:
                return [
                    SimpleNamespace(
                        text=graph_ctx.context_text,
                        id="graph_context",
                        score=1.0,
                        metadata={"source": "neo4j"},
                    )
                ]
        except Exception as exc:
            logger.warning("graph_rag tool failed: {}", exc)
        return []

    async def multi_source_rates(self, params: dict) -> list[Any]:
        try:
            rate_query = normalize_rate_query(
                rate_kinds=params.get("rate_kinds", [RateKind.MANDI_WHOLESALE.value]),
                commodity=params.get("commodity"),
                state=params.get("state", "Karnataka"),
                district=params.get("district"),
                market=params.get("market", params.get("location")),
                date=params.get("date"),
                include_reference=params.get("include_reference", True),
                force_live=params.get("force_live", False),
                comparison_depth=params.get("comparison_depth", ComparisonDepth.ALL_SOURCES.value),
            )
            service = await get_rate_service(agmarknet_api_key=get_agmarknet_api_key())
            result = await service.query(rate_query)
            documents: list[Any] = []
            for rate in result.canonical_rates:
                price = rate.modal_price or rate.price_value or rate.max_price or rate.min_price
                text = (
                    f"{rate.rate_kind.value} for {rate.commodity or rate.location_label} in "
                    f"{rate.location_label}: Rs.{price} {rate.unit} from {rate.source} on {rate.price_date}."
                )
                documents.append(
                    SimpleNamespace(
                        text=text,
                        id=f"rate_{rate.rate_kind.value}_{rate.location_label.lower()}",
                        score=1.0,
                        metadata={
                            "source": rate.source,
                            "rate_kind": rate.rate_kind.value,
                            "location_label": rate.location_label,
                            "price_date": rate.price_date.isoformat(),
                        },
                    )
                )
            return documents
        except Exception as exc:
            logger.warning("multi_source_rates tool failed: {}", exc)
            return []

    async def price_api(self, params: dict) -> list[Any]:
        """Backward-compatible mandi-only adapter over multi_source_rates."""
        price_params = {
            **params,
            "rate_kinds": [RateKind.MANDI_WHOLESALE.value],
            "market": params.get("market", params.get("location")),
        }
        return await self.multi_source_rates(price_params)

    async def weather_api(self, params: dict) -> list[Any]:
        if self.weather_client is None:
            return []
        try:
            location = params.get("location", params.get("district", ""))
            weather_data = await self.weather_client.get_forecast(location=location, days=params.get("days", 5))
            if weather_data:
                return [
                    SimpleNamespace(
                        text=f"Weather forecast for {location}: {weather_data}",
                        id="weather_api",
                        score=1.0,
                        metadata={"source": "imd"},
                    )
                ]
        except Exception as exc:
            logger.warning("weather_api tool failed: {}", exc)
        return []

    async def browser_scrape(self, params: dict) -> list[Any]:
        if self.browser_rag is None:
            return []
        try:
            return await self.browser_rag.retrieve_live(query=params.get("query", ""))
        except Exception as exc:
            logger.warning("browser_scrape tool failed: {}", exc)
            return []

    async def direct_llm(self, params: dict) -> list[Any]:
        logger.debug("direct_llm tool requested with params={}", params)
        return []
