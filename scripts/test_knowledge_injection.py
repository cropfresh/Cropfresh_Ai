"""
Test Real-Time Knowledge Injection (Phase 6)
=============================================
Tests for News Streamer, Market Alerts, Weather Advisory, and Scheme Crawler.

Run with:
    uv run python scripts/test_knowledge_injection.py

Author: CropFresh AI Team
"""

import asyncio
from typing import Dict, Any

from src.rag.knowledge_injection import (
    KnowledgeInjector, 
    NewsStreamer,
    MarketAlertSystem,
    WeatherAdvisorySystem,
    SchemeCrawler,
    RealTimeUpdate,
    AlertSeverity
)


async def test_news_streamer():
    """Test fetching and normalization of news."""
    print("Testing NewsStreamer...")
    streamer = NewsStreamer()
    news_items = await streamer.fetch_latest_news(limit=3)
    
    assert len(news_items) > 0
    print(f"Fetched {len(news_items)} news items.")
    print(f"Latest: {news_items[0].title}")
    
    # Check normalization
    assert isinstance(news_items[0], RealTimeUpdate)
    assert news_items[0].type == "news"


async def test_market_alert_system():
    """Test price alert generation logic."""
    print("\nTesting MarketAlertSystem...")
    alert_system = MarketAlertSystem(threshold_pct=10.0)
    
    # Test significant surge
    alert = await alert_system.check_price_alerts("Tomato", 150, 100)
    assert alert is not None
    assert alert.severity == AlertSeverity.CRITICAL
    assert "surged by 50.0%" in alert.title
    print(f"Generated alert: {alert.title}")
    
    # Test stable price (no alert)
    no_alert = await alert_system.check_price_alerts("Onion", 105, 100)
    assert no_alert is None
    print("Correctly ignored stable price (5% deviation < 10% threshold).")


async def test_weather_advisory():
    """Test weather-based advisory generation."""
    print("\nTesting WeatherAdvisorySystem...")
    sys = WeatherAdvisorySystem()
    
    # Test Tomato High Temp & Rain
    obs = {"temp": 38, "rain": 25, "humidity": 70}
    advisories = await sys.generate_advisories("Tomato", obs)
    
    assert len(advisories) >= 2
    titles = [a.title for a in advisories]
    assert any("High Temp" in t for t in titles)
    assert any("Heavy Rain" in t for t in titles)
    
    print(f"Generated {len(advisories)} tomato advisories.")
    print(f"Example: {advisories[0].title}")


async def test_scheme_crawler():
    """Test scheme crawler."""
    print("\nTesting SchemeCrawler...")
    crawler = SchemeCrawler()
    updates = await crawler.check_for_updates()
    
    assert len(updates) > 0
    assert updates[0].type == "scheme"
    print(f"Found {len(updates)} scheme updates.")
    print(f"Latest: {updates[0].title}")


async def test_knowledge_injection():
    """Test injecting updates into context."""
    print("\nTesting KnowledgeInjector...")
    injector = KnowledgeInjector()
    
    updates = [
        RealTimeUpdate(
            type="news",
            title="Heavy rain in Kolar",
            content="Forecast suggests 50mm rain.",
            source="Test",
            entities=["Kolar", "Rain"]
        )
    ]
    
    await injector.ingest_updates(updates)
    
    # Query matching entity
    context = injector.get_context_injection("Is there rain in Kolar?")
    assert "Heavy rain in Kolar" in context
    print("Injection Context Generated:\n", context)
    
    # Query mismatch
    empty_context = injector.get_context_injection("price of bitcoin")
    assert empty_context == ""
    print("Correctly returned empty context for irrelevant query.")
    
    # Test full pipeline
    print("\nTesting full injection pipeline...")
    count = await injector.fetch_all_updates()
    assert count > 0
    print(f"Fetched and ingested {count} total updates from all sources.")


async def main():
    print("=== Phase 6: Real-Time Knowledge Injection Tests ===")
    await test_news_streamer()
    await test_market_alert_system()
    await test_weather_advisory()
    await test_scheme_crawler()
    await test_knowledge_injection()
    print("\nAll knowledge injection tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
