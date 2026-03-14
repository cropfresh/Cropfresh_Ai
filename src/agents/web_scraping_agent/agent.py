"""
Web Scraping Agent
==================
Combines all mixins into a single cohesive Agent interface.
"""

from .extractor import LLMExtractorMixin
from .models import ScrapingResult


class WebScrapingAgent(LLMExtractorMixin):
    """
    Agent for intelligent web scraping using Playwright.
    
    Usage:
        agent = WebScrapingAgent()
        await agent.initialize()
        
        # Simple markdown extraction
        result = await agent.scrape_to_markdown("https://example.com")
        
        # Structured data extraction with LLM
        result = await agent.scrape_with_schema(
            url="https://agmarknet.gov.in/...",
            schema=MandiPriceSchema,
            instruction="Extract all mandi prices from the table"
        )
        
        await agent.close()
    """
    pass


async def scrape_url(url: str, to_markdown: bool = True) -> ScrapingResult:
    """
    Convenience function for one-off URL scraping.
    
    Args:
        url: URL to scrape
        to_markdown: If True, returns markdown; otherwise returns HTML
        
    Returns:
        ScrapingResult
    """
    agent = WebScrapingAgent()
    try:
        await agent.initialize()
        result = await agent.scrape_to_markdown(url)
        return result
    finally:
        await agent.close()
