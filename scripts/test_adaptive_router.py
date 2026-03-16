"""
Test: Adaptive Query Router — 30 query classification test
==========================================================
Validates the 8-strategy router rule-based pre-filter.
No external services needed — runs with USE_ADAPTIVE_ROUTER=true overridden.

Usage:
    uv run python scripts/test_adaptive_router.py
"""

import asyncio
import os
import sys

# Force enable the adaptive router for tests
os.environ["USE_ADAPTIVE_ROUTER"] = "true"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.query_analyzer import AdaptiveQueryRouter, RetrievalRoute

# ── Test cases: (query, has_image, expected_route) ───────────────────────────
TEST_CASES = [
    # DIRECT_LLM
    ("Hello, what is CropFresh?", False, RetrievalRoute.DIRECT_LLM),
    ("hi", False, RetrievalRoute.DIRECT_LLM),
    ("thanks, that was helpful", False, RetrievalRoute.DIRECT_LLM),
    ("namaste!", False, RetrievalRoute.DIRECT_LLM),
    ("who are you and what can you do", False, RetrievalRoute.DIRECT_LLM),

    # MULTIMODAL
    ("photo of my tomato plant has yellow leaves", False, RetrievalRoute.MULTIMODAL),
    ("image of diseased crop", False, RetrievalRoute.MULTIMODAL),
    ("check this crop photo", True, RetrievalRoute.MULTIMODAL),
    ("look at this leaf [image attached]", False, RetrievalRoute.MULTIMODAL),

    # LIVE_PRICE_API
    ("tomato price today in Hubli", False, RetrievalRoute.LIVE_PRICE_API),
    ("what is the current price of onion", False, RetrievalRoute.LIVE_PRICE_API),
    ("aaj mandi bhav kya hai", False, RetrievalRoute.LIVE_PRICE_API),
    ("live mandi rate for potato", False, RetrievalRoute.LIVE_PRICE_API),
    ("real-time bhaav for cotton", False, RetrievalRoute.LIVE_PRICE_API),

    # WEATHER_API
    ("will it rain tomorrow in Karnataka?", False, RetrievalRoute.WEATHER_API),
    ("weather forecast for next week", False, RetrievalRoute.WEATHER_API),
    ("monsoon prediction this year", False, RetrievalRoute.WEATHER_API),
    ("temperature forecast for Kolar", False, RetrievalRoute.WEATHER_API),
    ("barish kab hogi?", False, RetrievalRoute.WEATHER_API),

    # VECTOR_ONLY (rule-based fallback — should not match pre-filter shortcuts)
    ("how to grow tomatoes in Karnataka?", False, None),     # Goes to LLM/fallback
    ("what is organic farming?", False, None),               # Similar
    ("best pest control for rice", False, None),             # Similar

    # GRAPH_TRAVERSAL (rule-based fallback)
    ("which farmers in Kolar grow tomatoes?", False, RetrievalRoute.GRAPH_TRAVERSAL),
    ("show me farmers growing onions near Belgaum", False, RetrievalRoute.GRAPH_TRAVERSAL),

    # FULL_AGENTIC (complex decision)
    ("should I sell my tomatoes now or wait 2 weeks?", False, RetrievalRoute.FULL_AGENTIC),
    ("compare mango vs banana profitability", False, RetrievalRoute.FULL_AGENTIC),

    # BROWSER_SCRAPE (live scheme/news)
    ("latest scheme for tomato farmers 2026", False, RetrievalRoute.BROWSER_SCRAPE),
    ("new government policy for agriculture", False, RetrievalRoute.BROWSER_SCRAPE),
    ("news about pesticide ban in India", False, RetrievalRoute.BROWSER_SCRAPE),
]


async def run_tests():
    """Run all routing test cases and print a summary table."""
    router = AdaptiveQueryRouter(llm=None)  # Rule-based only (no LLM in test)

    print("\n" + "=" * 80)
    print("ADAPTIVE QUERY ROUTER — 30 Query Test Suite")
    print("=" * 80)
    print(f"{'STATUS':<8} {'EXPECTED':<22} {'GOT':<22} {'PRE_FILTER':<12} QUERY")
    print("─" * 100)

    passed = 0
    failed = 0
    skipped = 0

    for query, has_image, expected_route in TEST_CASES:
        decision = await router.route(query, has_image=has_image)
        got = decision.strategy

        if expected_route is None:
            # Not checking specific route — just verify it returns something valid
            status = "⬜ SKIP"
            skipped += 1
        elif got == expected_route:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1

        pre = "✓" if decision.pre_filter_matched else "✗"
        expected_str = expected_route.value if expected_route else "(any)"
        print(f"{status:<8} {expected_str:<22} {got.value:<22} {pre:<12} {query[:50]}")

    print("─" * 100)
    print(f"\nResults: ✅ {passed} passed | ❌ {failed} failed | ⬜ {skipped} skipped")
    print(f"Pass rate: {passed / (passed + failed) * 100:.1f}% (of definite assertions)\n")

    # Print cost stats
    print("Cost Summary (by strategy):")
    from src.rag.query_analyzer import ROUTE_COST_MAP
    for route, cost in ROUTE_COST_MAP.items():
        print(f"  {route.value:<22} ₹{cost:.3f}/query")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
