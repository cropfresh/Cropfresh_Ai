"""
Deep Research Map-Reduce
========================
Extracts facts from pages and synthesis into a final answer.
"""

import asyncio
import httpx
from loguru import logger

from .models import PageContent, ExtractedFact
from .constants import GROQ_FAST_MODEL, GROQ_SYNTH_MODEL
from .llm import _groq_complete


async def _extract_facts(
    page: PageContent,
    query: str,
    client: httpx.AsyncClient,
    api_key: str,
) -> ExtractedFact:
    """
    Ask a fast LLM to extract only query-relevant facts from a page (Map step).
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
    """Run the Map step across all pages concurrently."""
    tasks = [_extract_facts(p, query, client, api_key) for p in pages]
    results: list[ExtractedFact] = await asyncio.gather(*tasks)
    useful = [r for r in results if not r.skipped]
    logger.info(f"Map step: {len(useful)}/{len(results)} pages had useful facts")
    return results


async def synthesise_answer(
    query: str,
    facts: list[ExtractedFact],
    client: httpx.AsyncClient,
    api_key: str,
) -> str:
    """
    Synthesise a comprehensive answer from extracted facts (Reduce step).
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
