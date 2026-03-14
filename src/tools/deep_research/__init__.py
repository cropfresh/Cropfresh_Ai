"""
Deep Research Tool Package
==========================
Agentic Map-Reduce pipeline for comprehensive research.
"""

from .fetching import fetch_all_pages
from .map_reduce import extract_all_facts, synthesise_answer
from .models import DeepResearchResult, ExtractedFact, PageContent
from .tool import DeepResearchTool

__all__ = [
    "PageContent",
    "ExtractedFact",
    "DeepResearchResult",
    "DeepResearchTool",
    "fetch_all_pages",
    "extract_all_facts",
    "synthesise_answer",
]

from loguru import logger

from src.tools.registry import get_tool_registry

# ---------------------------------------------------------------------------
# Tool registry entry point
# ---------------------------------------------------------------------------

async def _deep_research(query: str, max_pages: int = 12) -> dict:
    """
    Perform deep research by fetching and analysing up to 15 websites.
    """
    tool = DeepResearchTool()
    result = await tool.research(query, max_pages=max_pages)
    return {
        "query": result.query,
        "answer": result.answer,
        "sources": result.sources,
        "pages_fetched": result.pages_fetched,
        "pages_useful": result.pages_useful,
        "formatted": tool.format_for_llm(result),
    }

# Auto-register when module is imported
try:
    _registry = get_tool_registry()
    _registry.add_tool(
        _deep_research,
        name="deep_research",
        description=(
            "Deep research tool that simultaneously fetches 10-15 websites, "
            "extracts relevant facts from each using AI, and synthesises a "
            "comprehensive multi-source answer with citations. Use this for "
            "complex comparative questions, market trend analysis, policy "
            "research, or any query needing verified multi-source data."
        ),
        category="web",
    )
    logger.debug("deep_research tool registered")
except Exception as _exc:
    logger.debug(f"Deep research tool registration deferred: {_exc}")
