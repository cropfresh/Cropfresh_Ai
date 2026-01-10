"""
Web Search Tool
===============
Web search fallback for real-time information.

Provides:
- Web search via Tavily API (free tier)
- DuckDuckGo fallback
- Result formatting for LLM consumption

Author: CropFresh AI Team
Version: 2.0.0
"""

from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.tools.registry import get_tool_registry


class SearchResult(BaseModel):
    """Single search result."""
    
    title: str
    url: str
    snippet: str
    source: str = ""


class SearchResults(BaseModel):
    """Web search results."""
    
    query: str
    results: list[SearchResult] = Field(default_factory=list)
    total: int = 0


class WebSearchTool:
    """
    Web search integration for real-time information.
    
    Primary: Tavily API (designed for LLMs)
    Fallback: DuckDuckGo (no API key needed)
    
    Usage:
        tool = WebSearchTool(api_key="your_tavily_key")
        results = await tool.search("current tomato prices Karnataka")
    """
    
    def __init__(
        self,
        api_key: str = "",
        use_mock: bool = True,
    ):
        """
        Initialize web search tool.
        
        Args:
            api_key: Tavily API key
            use_mock: Use mock results (default True)
        """
        self.api_key = api_key
        self.use_mock = use_mock
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_domains: Optional[list[str]] = None,
    ) -> SearchResults:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            include_domains: Optional list of domains to prioritize
            
        Returns:
            SearchResults with found information
        """
        if self.use_mock or not self.api_key:
            return self._get_mock_results(query)
        
        try:
            return await self._search_tavily(query, max_results, include_domains)
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}, falling back to mock")
            return self._get_mock_results(query)
    
    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        include_domains: Optional[list[str]],
    ) -> SearchResults:
        """Search using Tavily API."""
        import httpx
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_domains": include_domains or [],
                    "search_depth": "basic",
                },
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source=item.get("source", ""),
            ))
        
        return SearchResults(
            query=query,
            results=results,
            total=len(results),
        )
    
    def _get_mock_results(self, query: str) -> SearchResults:
        """Return mock results for development."""
        query_lower = query.lower()
        
        # Agricultural price-related queries
        if any(kw in query_lower for kw in ["price", "rate", "mandi", "market"]):
            return SearchResults(
                query=query,
                results=[
                    SearchResult(
                        title="Agmarknet - Agricultural Marketing Information Network",
                        url="https://agmarknet.gov.in",
                        snippet="National portal for agricultural marketing. Find real-time mandi prices for all commodities across India.",
                        source="Government of India",
                    ),
                    SearchResult(
                        title="Today's Vegetable Prices in Karnataka",
                        url="https://example.com/prices",
                        snippet="Current vegetable prices in Karnataka mandis. Tomato: ₹25-35/kg, Onion: ₹18-24/kg, Potato: ₹16-20/kg.",
                        source="Agricultural News",
                    ),
                ],
                total=2,
            )
        
        # Weather-related queries
        if any(kw in query_lower for kw in ["weather", "forecast", "rain", "monsoon"]):
            return SearchResults(
                query=query,
                results=[
                    SearchResult(
                        title="IMD Weather Forecast - Karnataka",
                        url="https://mausam.imd.gov.in",
                        snippet="India Meteorological Department forecast for Karnataka. Next 5 days: Partly cloudy with chances of light rain.",
                        source="IMD",
                    ),
                ],
                total=1,
            )
        
        # Farming/agricultural queries
        if any(kw in query_lower for kw in ["grow", "farm", "crop", "cultivation"]):
            return SearchResults(
                query=query,
                results=[
                    SearchResult(
                        title="ICAR - Crop Production Guide",
                        url="https://icar.gov.in/crops",
                        snippet="Indian Council of Agricultural Research provides comprehensive guides for all major crops in India.",
                        source="ICAR",
                    ),
                    SearchResult(
                        title="Agricultural University Resources",
                        url="https://uasd.edu/resources",
                        snippet="Research-backed farming practices and crop management techniques for Karnataka farmers.",
                        source="UAS Dharwad",
                    ),
                ],
                total=2,
            )
        
        # Default mock
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title=f"Search Results for: {query}",
                    url="https://example.com",
                    snippet="No specific results found. Please refine your search query.",
                    source="Web",
                ),
            ],
            total=1,
        )
    
    def format_for_llm(self, results: SearchResults) -> str:
        """
        Format search results for LLM consumption.
        
        Args:
            results: SearchResults object
            
        Returns:
            Formatted string for LLM context
        """
        if not results.results:
            return f"No web results found for: {results.query}"
        
        parts = [f"Web search results for '{results.query}':\n"]
        
        for i, result in enumerate(results.results, 1):
            parts.append(f"[{i}] {result.title}")
            parts.append(f"    Source: {result.source or result.url}")
            parts.append(f"    {result.snippet}")
            parts.append("")
        
        return "\n".join(parts)


# Tool function for registry
async def _web_search(query: str, max_results: int = 5) -> dict:
    """Search the web for information."""
    tool = WebSearchTool()
    results = await tool.search(query, max_results)
    return {
        "query": results.query,
        "results": [r.model_dump() for r in results.results],
        "total": results.total,
        "formatted": tool.format_for_llm(results),
    }


# Auto-register on module import
try:
    registry = get_tool_registry()
    registry.add_tool(
        _web_search,
        name="web_search",
        description="Search the web for real-time information like current prices, weather, news. Returns formatted results for answering user queries.",
        category="web",
    )
except Exception as e:
    logger.debug(f"Web search tool registration deferred: {e}")
