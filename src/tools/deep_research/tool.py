"""
Deep Research Tool Core
=======================
Agentic 5-step Deep Research pipeline orchestrator.
"""

import os
from typing import Optional

import httpx
from loguru import logger

from src.tools.web_search import WebSearchTool

from .constants import GROQ_API_KEY_ENV, MAX_PAGES
from .fetching import fetch_all_pages
from .map_reduce import extract_all_facts, synthesise_answer
from .models import DeepResearchResult


class DeepResearchTool:
    """
    Agentic 5-step Deep Research pipeline.

    Simultaneously fetches 10-15 web pages, extracts facts from each
    using a fast LLM, then synthesises a single comprehensive answer.
    """

    def __init__(
        self,
        api_key: str = "",
        search_api_key: str = "",
        max_pages: int = MAX_PAGES,
    ):
        self.api_key = api_key or os.getenv(GROQ_API_KEY_ENV, "")
        self.search_api_key = search_api_key or os.getenv("TAVILY_API_KEY", "")
        self.max_pages = min(max_pages, MAX_PAGES)
        self._search_tool = WebSearchTool(api_key=self.search_api_key)

    async def research(
        self,
        query: str,
        max_pages: Optional[int] = None,
    ) -> DeepResearchResult:
        """Execute the full 5-step deep research pipeline."""
        limit = max_pages or self.max_pages
        logger.info(f"Deep Research started: '{query}' | max_pages={limit}")

        search_results = await self._search_tool.search(query, max_results=limit)
        urls = [r.url for r in search_results.results if r.url]

        if not urls:
            logger.warning("No URLs returned by search tool")
            return DeepResearchResult(
                query=query,
                answer="No search results found. Please check your query.",
            )

        async with httpx.AsyncClient() as client:
            # Step 3 — Fetch all pages simultaneously
            pages = await fetch_all_pages(urls, client)

            # Step 4 — Map: extract facts per page simultaneously
            all_facts = await extract_all_facts(pages, query, client, self.api_key)

        useful_facts = [f for f in all_facts if not f.skipped]

        # Step 5 — Reduce: synthesise final answer
        async with httpx.AsyncClient() as client:
            answer = await synthesise_answer(query, useful_facts, client, self.api_key)

        sources = [f.url for f in useful_facts]

        return DeepResearchResult(
            query=query,
            answer=answer,
            sources=sources,
            pages_fetched=len(pages),
            pages_useful=len(useful_facts),
        )

    def format_for_llm(self, result: DeepResearchResult) -> str:
        """Format a DeepResearchResult as a string for LLM context injection."""
        lines = [
            f"Deep Research: '{result.query}'",
            f"Pages searched: {result.pages_fetched} | "
            f"Useful: {result.pages_useful}",
            "",
            result.answer,
            "",
            "Sources:",
        ]
        for i, src in enumerate(result.sources, 1):
            lines.append(f"  [{i}] {src}")
        return "\n".join(lines)
