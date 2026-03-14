"""
Deep Research Fetching
======================
Concurrent fetching of web pages via Jina Reader.
"""

import asyncio
import httpx
from loguru import logger

from .models import PageContent
from .constants import JINA_BASE_URL, FETCH_TIMEOUT_SEC, MAX_CONTENT_CHARS, MAX_PAGES


async def _fetch_page(url: str, client: httpx.AsyncClient) -> PageContent:
    """Fetch a web page and convert it to clean Markdown via Jina Reader."""
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


async def fetch_all_pages(urls: list[str], client: httpx.AsyncClient) -> list[PageContent]:
    """Fetch all pages concurrently (Step 3 — Fetch)."""
    tasks = [_fetch_page(url, client) for url in urls[:MAX_PAGES]]
    results: list[PageContent] = await asyncio.gather(*tasks, return_exceptions=False)
    success_count = sum(1 for r in results if r.success)
    logger.info(f"Fetched {success_count}/{len(results)} pages successfully")
    return results
