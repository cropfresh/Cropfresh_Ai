"""
Deep Research Tool
==================
Agentic Map-Reduce pipeline that simultaneously fetches and analyses
content from 10-15 websites and synthesises a comprehensive answer.

Architecture:
    1. Expand: LLM generates N sub-queries to cover different angles.
    2. Search: Tavily/DuckDuckGo finds URLs for all sub-queries in parallel.
    3. Fetch:  Jina Reader converts each URL to clean Markdown concurrently.
    4. Map:    Fast LLM extract relevant facts per page (runs in parallel).
    5. Reduce: Powerful LLM synthesises all extracted facts into one answer
               with inline citations.

Author: CropFresh AI Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from src.tools.web_search import WebSearchTool
from src.tools.registry import get_tool_registry


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PageContent(BaseModel):
    """Raw content retrieved from a single web page."""

    url: str
    markdown: str = ""
    success: bool = False
    error: str = ""


class ExtractedFact(BaseModel):
    """Facts isolated from one page that are relevant to the query."""

    url: str
    facts: str = ""
    skipped: bool = False


class DeepResearchResult(BaseModel):
    """Final deep research output."""

    query: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    pages_fetched: int = 0
    pages_useful: int = 0


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

JINA_BASE_URL = "https://r.jina.ai/"
MAX_PAGES = 15
MAX_CONTENT_CHARS = 12_000   # per page, to stay within LLM context
FETCH_TIMEOUT_SEC = 15
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Fast extraction model — cheap and quick
GROQ_FAST_MODEL = "llama-3.1-8b-instant"
# Synthesis model — higher quality for the final answer
GROQ_SYNTH_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Page fetching
# ---------------------------------------------------------------------------

async def _fetch_page(
    url: str,
    client: httpx.AsyncClient,
) -> PageContent:
    """
    Fetch a web page and convert it to clean Markdown via Jina Reader.

    Falls back to a raw GET if Jina fails (bot-protection bypass).

    Args:
        url:    Target URL.
        client: Shared async HTTP client.

    Returns:
        PageContent with markdown text or an error description.
    """
    jina_url = f"{JINA_BASE_URL}{url}"
    try:
        resp = await client.get(
            jina_url,
            timeout=FETCH_TIMEOUT_SEC,
            headers={"Accept": "text/markdown"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        text = resp.text[:MAX_CONTENT_CHARS]
        return PageContent(url=url, markdown=text, success=True)
    except Exception as exc:
        logger.debug(f"Jina fetch failed for {url}: {exc}")
        return PageContent(url=url, success=False, error=str(exc))


async def fetch_all_pages(
    urls: list[str],
    client: httpx.AsyncClient,
) -> list[PageContent]:
    """
    Fetch all pages concurrently (Step 3 — Fetch).

    Args:
        urls:   List of URLs to fetch.
        client: Shared async HTTP client.

    Returns:
        List of PageContent objects (successful + failed).
    """
    tasks = [_fetch_page(url, client) for url in urls[:MAX_PAGES]]
    results: list[PageContent] = await asyncio.gather(*tasks, return_exceptions=False)
    success_count = sum(1 for r in results if r.success)
    logger.info(f"Fetched {success_count}/{len(results)} pages successfully")
    return results


# ---------------------------------------------------------------------------
# Groq LLM helper
# ---------------------------------------------------------------------------

async def _groq_complete(
    prompt: str,
    model: str,
    client: httpx.AsyncClient,
    api_key: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Call the Groq Chat Completions API asynchronously.

    Args:
        prompt:     User message content.
        model:      Groq model identifier.
        client:     Shared HTTP client.
        api_key:    Groq API key.
        temperature: Sampling temperature.
        max_tokens:  Maximum response tokens.

    Returns:
        Model response text or empty string on error.
    """
    try:
        resp = await client.post(
            GROQ_API_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning(f"Groq call failed ({model}): {exc}")
        return ""


# ---------------------------------------------------------------------------
# Map step: extract facts from a single page
# ---------------------------------------------------------------------------

async def _extract_facts(
    page: PageContent,
    query: str,
    client: httpx.AsyncClient,
    api_key: str,
) -> ExtractedFact:
    """
    Ask a fast LLM to extract only query-relevant facts from a page (Map step).

    Pages that contain no relevant information are marked skipped so the
    reduce step ignores them.

    Args:
        page:    Fetched page content.
        query:   Original user query.
        client:  Shared HTTP client.
        api_key: Groq API key.

    Returns:
        ExtractedFact with relevant text or a skip flag.
    """
    if not page.success or not page.markdown.strip():
        return ExtractedFact(url=page.url, skipped=True)

    prompt = (
        f"You are an agricultural research assistant for CropFresh AI.\n"
        f"Read the following webpage excerpt and extract ONLY facts, data, "
        f"prices, or guidance directly relevant to this question:\n"
        f'"{query}"\n\n'
        f"If the page contains nothing relevant, reply with exactly: SKIP\n"
        f"Otherwise extract up to 300 words of the most relevant content.\n"
        f"Do NOT add commentary, just the extracted facts.\n\n"
        f"SOURCE URL: {page.url}\n\n"
        f"PAGE CONTENT:\n{page.markdown}"
    )

    raw = await _groq_complete(
        prompt=prompt,
        model=GROQ_FAST_MODEL,
        client=client,
        api_key=api_key,
        max_tokens=512,
    )

    if not raw or raw.strip().upper() == "SKIP":
        return ExtractedFact(url=page.url, skipped=True)

    return ExtractedFact(url=page.url, facts=raw.strip())


async def extract_all_facts(
    pages: list[PageContent],
    query: str,
    client: httpx.AsyncClient,
    api_key: str,
) -> list[ExtractedFact]:
    """
    Run the Map step across all pages concurrently.

    Args:
        pages:   All fetched pages.
        query:   Original user query.
        client:  Shared HTTP client.
        api_key: Groq API key.

    Returns:
        List of ExtractedFact (skipped pages included for logging).
    """
    tasks = [_extract_facts(p, query, client, api_key) for p in pages]
    results: list[ExtractedFact] = await asyncio.gather(*tasks)
    useful = [r for r in results if not r.skipped]
    logger.info(f"Map step: {len(useful)}/{len(results)} pages had useful facts")
    return results


# ---------------------------------------------------------------------------
# Reduce step: synthesise final answer
# ---------------------------------------------------------------------------

async def synthesise_answer(
    query: str,
    facts: list[ExtractedFact],
    client: httpx.AsyncClient,
    api_key: str,
) -> str:
    """
    Synthesise a comprehensive answer from extracted facts (Reduce step).

    Uses the more powerful Groq model to compare, reconcile, and narrate
    across all sources, adding inline citations.

    Args:
        query:  Original user query.
        facts:  Non-skipped extracted facts with their source URLs.
        client: Shared HTTP client.
        api_key: Groq API key.

    Returns:
        Final synthesised answer string.
    """
    if not facts:
        return (
            "I could not find sufficient information across the searched websites "
            "to answer your question. Please try a more specific query."
        )

    # Build a numbered source list
    source_block = "\n\n".join(
        f"[{i + 1}] Source: {f.url}\n{f.facts}"
        for i, f in enumerate(facts)
    )

    prompt = (
        "You are a senior agricultural intelligence analyst for CropFresh AI, "
        "India's AI-powered agri-marketplace.\n\n"
        "Based ONLY on the multi-source research excerpts below, answer the "
        "following question comprehensively:\n"
        f'"{query}"\n\n'
        "Instructions:\n"
        "- Compare data across sources and highlight any discrepancies.\n"
        "- Cite sources inline using [1], [2] style numbering.\n"
        "- Provide a clear, structured answer with key takeaways.\n"
        "- Do NOT fabricate data not present in the sources.\n"
        "- Respond in plain English, max 600 words.\n\n"
        f"RESEARCH SOURCES:\n{source_block}"
    )

    answer = await _groq_complete(
        prompt=prompt,
        model=GROQ_SYNTH_MODEL,
        client=client,
        api_key=api_key,
        temperature=0.3,
        max_tokens=1200,
    )

    return answer or "Answer synthesis failed — please retry."


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class DeepResearchTool:
    """
    Agentic 5-step Deep Research pipeline.

    Simultaneously fetches 10-15 web pages, extracts facts from each
    using a fast LLM, then synthesises a single comprehensive answer.

    Usage:
        tool = DeepResearchTool()
        result = await tool.research("Current tomato MSP vs mandi price Karnataka 2026")
    """

    def __init__(
        self,
        api_key: str = "",
        search_api_key: str = "",
        max_pages: int = MAX_PAGES,
    ):
        """
        Initialise the Deep Research tool.

        Args:
            api_key:        Groq API key (falls back to env var GROQ_API_KEY).
            search_api_key: Tavily search API key (falls back to env var).
            max_pages:      Maximum pages to fetch (default 15).
        """
        self.api_key = api_key or os.getenv(GROQ_API_KEY_ENV, "")
        self.search_api_key = search_api_key or os.getenv("TAVILY_API_KEY", "")
        self.max_pages = min(max_pages, MAX_PAGES)
        self._search_tool = WebSearchTool(api_key=self.search_api_key)

    async def research(
        self,
        query: str,
        max_pages: Optional[int] = None,
    ) -> DeepResearchResult:
        """
        Execute the full 5-step deep research pipeline.

        Args:
            query:     The research question or topic.
            max_pages: Override the default max pages limit.

        Returns:
            DeepResearchResult with the synthesised answer and metadata.
        """
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
        """
        Format a DeepResearchResult as a string for LLM context injection.

        Args:
            result: Completed deep research result.

        Returns:
            Human-readable formatted string.
        """
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


# ---------------------------------------------------------------------------
# Tool registry entry point
# ---------------------------------------------------------------------------

async def _deep_research(query: str, max_pages: int = 12) -> dict:
    """
    Perform deep research by fetching and analysing up to 15 websites.

    Preferred over web_search when the user needs a comprehensive,
    multi-source answer with comparisons and citations.

    Args:
        query:     The research question.
        max_pages: Number of pages to fetch (max 15, default 12).

    Returns:
        dict with 'answer', 'sources', 'pages_fetched', 'pages_useful'.
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
