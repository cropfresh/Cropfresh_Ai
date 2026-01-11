"""Quick test for Crawl4AI."""
import asyncio
from crawl4ai import AsyncWebCrawler

async def test():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://example.com")
        print(f"✅ Crawl4AI works! Content length: {len(result.markdown)}")
        print(f"   Title extracted: {result.markdown[:100]}...")

asyncio.run(test())
