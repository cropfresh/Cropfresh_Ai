"""
Test Deep Research Tool (Manual Script)
========================================
Run this script to verify the DeepResearchTool works end-to-end
in mock mode (no real API key required for basic smoke test).

Usage:
    uv run python scripts/test_deep_research.py
    uv run python scripts/test_deep_research.py --live  (requires GROQ_API_KEY + TAVILY_API_KEY)

Author: CropFresh AI Team
"""

from __future__ import annotations

import asyncio
import sys


async def run_mock_test() -> None:
    """Smoke test using mocked Groq and Jina calls."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.tools.deep_research import DeepResearchTool
    from src.tools.web_search import SearchResult, SearchResults

    tool = DeepResearchTool(api_key="mock-key")

    # Inject mock search results
    tool._search_tool.search = AsyncMock(return_value=SearchResults(
        query="test",
        results=[
            SearchResult(
                title="Agmarknet",
                url="https://agmarknet.gov.in/prices",
                snippet="Tomato price Karnataka",
                source="Gov",
            ),
            SearchResult(
                title="Krishijagran",
                url="https://krishijagran.com/tomato",
                snippet="Tomato market analysis",
                source="Media",
            ),
        ],
        total=2,
    ))

    mock_get = MagicMock()
    mock_get.text = "Tomato wholesale price in Kolar: ₹32/kg. Retail: ₹45/kg."
    mock_get.raise_for_status = MagicMock()

    mock_post = MagicMock()
    mock_post.raise_for_status = MagicMock()
    mock_post.json = MagicMock(return_value={
        "choices": [{
            "message": {
                "content": (
                    "Tomato prices in Karnataka range from ₹32–45/kg. "
                    "Kolar mandi reports ₹32/kg wholesale [1]. "
                    "Retail markets show ₹45/kg [2]. "
                    "Source data is consistent across platforms."
                )
            }
        }]
    })

    with patch("src.tools.deep_research.httpx.AsyncClient") as mock_cls:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.get = AsyncMock(return_value=mock_get)
        ctx.post = AsyncMock(return_value=mock_post)
        mock_cls.return_value = ctx

        result = await tool.research(
            "What is the current tomato wholesale and retail price in Karnataka?",
            max_pages=2,
        )

    print("=" * 60)
    print("MOCK TEST RESULT")
    print("=" * 60)
    print(tool.format_for_llm(result))
    print("=" * 60)
    print(f"✅ Mock test passed — pages_fetched={result.pages_fetched}, "
          f"pages_useful={result.pages_useful}")


async def run_live_test() -> None:
    """Live test hitting real Jina Reader and Groq APIs."""
    import os

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        print("❌ GROQ_API_KEY not set. Cannot run live test.")
        sys.exit(1)

    from src.tools.deep_research import DeepResearchTool

    tool = DeepResearchTool(api_key=groq_key)

    print("🔍 Running LIVE Deep Research...")
    print("Query: Current tomato MSP vs mandi price in Karnataka vs Maharashtra 2026")
    print("-" * 60)

    result = await tool.research(
        "Compare current tomato MSP and mandi market prices Karnataka vs Maharashtra 2026",
        max_pages=10,
    )

    print(tool.format_for_llm(result))
    print("-" * 60)
    print(f"✅ Live test done — pages_fetched={result.pages_fetched}, "
          f"pages_useful={result.pages_useful}")


if __name__ == "__main__":
    if "--live" in sys.argv:
        asyncio.run(run_live_test())
    else:
        asyncio.run(run_mock_test())
