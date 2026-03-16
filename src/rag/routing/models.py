from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries supported by the legacy query analyzer."""

    VECTOR_SEARCH = "vector"
    WEB_SEARCH = "web"
    DECOMPOSE = "decompose"
    DIRECT = "direct"


class QueryCategory(str, Enum):
    """Domain categories used for targeted retrieval."""

    AGRONOMY = "agronomy"
    MARKET = "market"
    PLATFORM = "platform"
    REGULATORY = "regulatory"
    GENERAL = "general"


class QueryAnalysis(BaseModel):
    """Legacy query analysis result."""

    original_query: str
    query_type: QueryType
    category: QueryCategory
    sub_queries: list[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.8
    crops: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    time_sensitive: bool = False


class RetrievalRoute(str, Enum):
    """8-strategy adaptive retrieval routes."""

    DIRECT_LLM = "direct_llm"
    VECTOR_ONLY = "vector_only"
    GRAPH_TRAVERSAL = "graph_traversal"
    LIVE_PRICE_API = "live_price_api"
    WEATHER_API = "weather_api"
    BROWSER_SCRAPE = "browser_scrape"
    MULTIMODAL = "multimodal"
    FULL_AGENTIC = "full_agentic"


ROUTE_COST_MAP: dict[RetrievalRoute, float] = {
    RetrievalRoute.DIRECT_LLM: 0.03,
    RetrievalRoute.VECTOR_ONLY: 0.12,
    RetrievalRoute.GRAPH_TRAVERSAL: 0.15,
    RetrievalRoute.LIVE_PRICE_API: 0.05,
    RetrievalRoute.WEATHER_API: 0.05,
    RetrievalRoute.BROWSER_SCRAPE: 0.25,
    RetrievalRoute.MULTIMODAL: 0.35,
    RetrievalRoute.FULL_AGENTIC: 0.55,
}


class RoutingDecision(BaseModel):
    """Adaptive router output."""

    strategy: RetrievalRoute
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    estimated_cost_inr: float
    entities: dict[str, list[str]] = Field(default_factory=dict)
    requires_live_data: bool = False
    requires_image: bool = False
    pre_filter_matched: bool = False
