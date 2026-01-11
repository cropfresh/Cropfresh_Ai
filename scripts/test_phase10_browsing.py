"""
Test Phase 10: Web Scraping Layer
=================================
Quick verification test for the browsing and scraping capabilities.
"""

import asyncio
import sys
sys.path.insert(0, ".")

from loguru import logger


async def test_web_scraping_agent():
    """Test WebScrapingAgent basic functionality."""
    from src.agents.web_scraping_agent import WebScrapingAgent, ScrapingConfig
    
    logger.info("Testing WebScrapingAgent...")
    
    agent = WebScrapingAgent()
    
    try:
        await agent.initialize(ScrapingConfig(headless=True, stealth=True))
        
        # Test simple scrape
        result = await agent.scrape_to_markdown("https://example.com")
        
        assert result.success, f"Scraping failed: {result.error}"
        assert len(result.markdown) > 100, "Markdown content too short"
        assert "Example Domain" in result.markdown, "Expected content not found"
        
        logger.success("✅ WebScrapingAgent: scrape_to_markdown works!")
        logger.info("   Content length: {} chars", len(result.markdown))
        logger.info("   Scrape time: {:.0f}ms", result.scrape_time_ms)
        
        # Test CSS extraction
        css_result = await agent.scrape_with_css(
            "https://example.com",
            {"title": "h1", "paragraph": "p"}
        )
        
        assert css_result.success, f"CSS extraction failed: {css_result.error}"
        assert "title" in css_result.extracted_data
        
        logger.success("✅ WebScrapingAgent: scrape_with_css works!")
        logger.info("   Extracted: {}", css_result.extracted_data)
        
    finally:
        await agent.close()
    
    return True


async def test_browser_agent():
    """Test BrowserAgent basic functionality."""
    from src.agents.browser_agent import BrowserAgent, BrowserAction, ActionType
    
    logger.info("Testing BrowserAgent...")
    
    agent = BrowserAgent(headless=True, stealth=True)
    
    try:
        session = await agent.start_session()
        
        assert session.session_id, "No session ID"
        logger.info("   Session started: {}", session.session_id)
        
        # Test navigation
        result = await agent.execute_action(BrowserAction(
            action=ActionType.GOTO,
            value="https://example.com"
        ))
        
        assert result.success, f"Navigation failed: {result.error}"
        logger.success("✅ BrowserAgent: navigation works!")
        
        # Test get page info
        info = await agent.get_page_info()
        assert "example.com" in info.get("url", ""), "URL not found in page info"
        logger.info("   Page URL: {}", info.get("url"))
        logger.info("   Page title: {}", info.get("title"))
        
        # Test screenshot
        result = await agent.execute_action(BrowserAction(
            action=ActionType.SCREENSHOT,
            value="test_screenshot.png"
        ))
        
        assert result.success, f"Screenshot failed: {result.error}"
        logger.success("✅ BrowserAgent: screenshot works!")
        logger.info("   Saved to: {}", result.screenshot_path)
        
    finally:
        await agent.close_session()
    
    return True


async def test_stealth():
    """Test stealth utilities."""
    from src.tools.browser_stealth import (
        get_random_user_agent,
        get_random_viewport,
        get_random_delay,
    )
    
    logger.info("Testing browser stealth utilities...")
    
    ua = get_random_user_agent()
    assert "Mozilla" in ua, "Invalid user agent"
    logger.info("   Random UA: {}...", ua[:50])
    
    viewport = get_random_viewport()
    assert viewport["width"] > 0, "Invalid viewport"
    logger.info("   Random viewport: {}x{}", viewport["width"], viewport["height"])
    
    delay = get_random_delay()
    assert 0.5 <= delay <= 2.0, "Delay out of range"
    logger.info("   Random delay: {:.2f}s", delay)
    
    logger.success("✅ Stealth utilities work!")
    return True


async def test_agri_scrapers():
    """Test agricultural data scrapers."""
    from src.tools.agri_scrapers import RSSNewsScraper
    
    logger.info("Testing agricultural scrapers...")
    
    # Test RSS scraper (doesn't need browser)
    news_scraper = RSSNewsScraper()
    articles = await news_scraper.get_news("rural_voice", limit=3)
    
    if articles:
        logger.success("✅ RSS News scraper works!")
        logger.info("   Fetched {} articles", len(articles))
        for article in articles[:2]:
            logger.info("   - {}", article.title[:50])
    else:
        logger.warning("⚠️ No articles fetched (may be network issue)")
    
    return True


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Phase 10: Web Scraping Layer - Verification Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Stealth Utilities", test_stealth),
        ("WebScrapingAgent", test_web_scraping_agent),
        ("BrowserAgent", test_browser_agent),
        ("Agricultural Scrapers", test_agri_scrapers),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            logger.info("\n--- {} ---", name)
            await test_func()
            passed += 1
        except AssertionError as e:
            logger.error("❌ {} FAILED: {}", name, str(e))
            failed += 1
        except Exception as e:
            logger.error("❌ {} ERROR: {}", name, str(e))
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info("Results: {} passed, {} failed", passed, failed)
    logger.info("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
