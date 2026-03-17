"""Research tool registration for agent-facing tool registries."""

from __future__ import annotations

from loguru import logger

from src.tools.deep_research import DeepResearchTool
from src.tools.registry import ToolRegistry
from src.tools.web_search import WebSearchTool


async def _deep_research(query: str, max_pages: int = 12) -> dict:
    result = await DeepResearchTool().research(query, max_pages=max_pages)
    return {
        "query": result.query,
        "answer": result.answer,
        "sources": result.sources,
        "pages_fetched": result.pages_fetched,
        "pages_useful": result.pages_useful,
    }


async def _web_search(query: str, max_results: int = 5) -> dict:
    results = await WebSearchTool().search(query, max_results)
    return {
        "query": results.query,
        "results": [item.model_dump(mode="json") for item in results.results],
        "total": results.total,
    }


def register_research_tools(registry: ToolRegistry) -> None:
    """Register web and deep-research tools."""
    try:
        registry.add_tool(
            _deep_research,
            name="deep_research",
            description="Run multi-source research and synthesis over a web query.",
            category="research",
        )
        registry.add_tool(
            _web_search,
            name="web_search",
            description="Search the web for current information and snippets.",
            category="research",
        )
    except Exception as exc:
        logger.debug("Research tool registration skipped: {}", exc)
