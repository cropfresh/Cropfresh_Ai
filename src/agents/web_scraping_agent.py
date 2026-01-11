"""
WebScrapingAgent
================
Intelligent web scraping with LLM-powered extraction using Playwright.

Capabilities:
- Scrape any URL and extract clean markdown content
- Extract structured data using Pydantic schemas via LLM
- Handle dynamic JavaScript-rendered content
- Full stealth mode with anti-detection
- CSS selector extraction (no LLM cost)

Author: CropFresh AI Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any, Optional, Type
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

# Local imports
from src.tools.browser_stealth import apply_stealth, get_random_user_agent, StealthConfig


class ScrapingResult(BaseModel):
    """Result from web scraping operation."""
    url: str
    success: bool
    markdown: str = ""
    html: str = ""
    extracted_data: dict = Field(default_factory=dict)
    error: Optional[str] = None
    cached: bool = False
    scrape_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class ScrapingConfig(BaseModel):
    """Configuration for scraping operations."""
    timeout: int = 30000  # ms
    wait_for_selector: Optional[str] = None
    wait_for_load_state: str = "networkidle"
    screenshot: bool = False
    full_page_screenshot: bool = True
    stealth: bool = True
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720


class WebScrapingAgent:
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
    
    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_minutes: int = 15,
    ):
        """
        Initialize WebScrapingAgent.
        
        Args:
            llm_provider: LLM provider for structured extraction
            cache_dir: Directory for caching scraped content
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.llm_provider = llm_provider
        self.cache_dir = cache_dir or Path("data/scraping_cache")
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("WebScrapingAgent initialized")
    
    async def initialize(self, config: Optional[ScrapingConfig] = None) -> None:
        """Initialize browser instance."""
        config = config or ScrapingConfig()
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=config.headless,
        )
        
        # Create context with stealth settings
        self._context = await self._browser.new_context(
            viewport={"width": config.viewport_width, "height": config.viewport_height},
            user_agent=get_random_user_agent() if config.stealth else None,
        )
        
        logger.info("Browser initialized (headless={})", config.headless)
    
    async def close(self) -> None:
        """Clean up browser resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")
    
    async def scrape_to_markdown(
        self,
        url: str,
        config: Optional[ScrapingConfig] = None,
        use_cache: bool = True,
    ) -> ScrapingResult:
        """
        Scrape URL and return clean markdown content.
        
        Args:
            url: URL to scrape
            config: Scraping configuration
            use_cache: Whether to use cached content if available
            
        Returns:
            ScrapingResult with markdown content
        """
        config = config or ScrapingConfig()
        start_time = datetime.now()
        
        # Check cache
        if use_cache:
            cached = self._get_cached(url)
            if cached:
                logger.debug("Cache hit for {}", url)
                return cached
        
        try:
            # Ensure browser is initialized
            if not self._browser:
                await self.initialize(config)
            
            page = await self._context.new_page()
            
            # Apply stealth if enabled
            if config.stealth:
                await apply_stealth(page)
            
            # Navigate to URL
            await page.goto(url, timeout=config.timeout, wait_until=config.wait_for_load_state)
            
            # Wait for specific selector if provided
            if config.wait_for_selector:
                await page.wait_for_selector(config.wait_for_selector, timeout=config.timeout)
            
            # Get page content
            html = await page.content()
            markdown = self._html_to_markdown(html)
            
            # Take screenshot if requested
            screenshot_path = None
            if config.screenshot:
                screenshot_path = self.cache_dir / f"{self._url_hash(url)}.png"
                await page.screenshot(path=str(screenshot_path), full_page=config.full_page_screenshot)
            
            await page.close()
            
            result = ScrapingResult(
                url=url,
                success=True,
                markdown=markdown,
                html=html,
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
            
            # Cache result
            if use_cache:
                self._set_cached(url, result)
            
            logger.info("Scraped {} ({:.0f}ms)", url, result.scrape_time_ms)
            return result
            
        except Exception as e:
            logger.error("Failed to scrape {}: {}", url, str(e))
            return ScrapingResult(
                url=url,
                success=False,
                error=str(e),
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
    
    async def scrape_with_css(
        self,
        url: str,
        selectors: dict[str, str],
        config: Optional[ScrapingConfig] = None,
    ) -> ScrapingResult:
        """
        Extract data using CSS selectors (no LLM cost).
        
        Args:
            url: URL to scrape
            selectors: Dict mapping field names to CSS selectors
            config: Scraping configuration
            
        Returns:
            ScrapingResult with extracted data
            
        Example:
            result = await agent.scrape_with_css(
                url="https://example.com",
                selectors={
                    "title": "h1.title",
                    "prices": "table.prices tr td:nth-child(2)",
                    "date": "span.date",
                }
            )
        """
        config = config or ScrapingConfig()
        start_time = datetime.now()
        
        try:
            if not self._browser:
                await self.initialize(config)
            
            page = await self._context.new_page()
            
            if config.stealth:
                await apply_stealth(page)
            
            await page.goto(url, timeout=config.timeout, wait_until=config.wait_for_load_state)
            
            extracted = {}
            for field_name, selector in selectors.items():
                try:
                    elements = await page.query_selector_all(selector)
                    if len(elements) == 1:
                        extracted[field_name] = await elements[0].inner_text()
                    else:
                        extracted[field_name] = [await el.inner_text() for el in elements]
                except Exception as e:
                    logger.warning("Failed to extract {}: {}", field_name, str(e))
                    extracted[field_name] = None
            
            await page.close()
            
            return ScrapingResult(
                url=url,
                success=True,
                extracted_data=extracted,
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
            
        except Exception as e:
            logger.error("CSS extraction failed for {}: {}", url, str(e))
            return ScrapingResult(
                url=url,
                success=False,
                error=str(e),
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
    
    async def scrape_with_schema(
        self,
        url: str,
        schema: Type[BaseModel],
        instruction: str,
        config: Optional[ScrapingConfig] = None,
    ) -> ScrapingResult:
        """
        Extract structured data using LLM and Pydantic schema.
        
        Args:
            url: URL to scrape
            schema: Pydantic model class for extraction
            instruction: Extraction instruction for LLM
            config: Scraping configuration
            
        Returns:
            ScrapingResult with extracted data matching schema
            
        Example:
            class MandiPrice(BaseModel):
                commodity: str
                price: float
                market: str
                
            result = await agent.scrape_with_schema(
                url="https://agmarknet.gov.in/...",
                schema=MandiPrice,
                instruction="Extract all commodity prices from the table"
            )
        """
        if not self.llm_provider:
            return ScrapingResult(
                url=url,
                success=False,
                error="LLM provider not configured for schema extraction",
            )
        
        # First scrape the page content
        markdown_result = await self.scrape_to_markdown(url, config)
        
        if not markdown_result.success:
            return markdown_result
        
        try:
            # Use LLM to extract structured data
            extraction_prompt = f"""Extract data from the following web page content according to the schema.

INSTRUCTION: {instruction}

SCHEMA:
{json.dumps(schema.model_json_schema(), indent=2)}

PAGE CONTENT:
{markdown_result.markdown[:15000]}  # Limit content to avoid token limits

Return a valid JSON object or array matching the schema. Only output JSON, no explanation."""

            # Call LLM (using the provider's interface)
            llm_response = await self.llm_provider.agenerate(extraction_prompt)
            
            # Parse JSON response
            json_match = re.search(r'[\[\{].*[\]\}]', llm_response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
            else:
                extracted = {}
            
            return ScrapingResult(
                url=url,
                success=True,
                markdown=markdown_result.markdown,
                extracted_data=extracted,
                scrape_time_ms=markdown_result.scrape_time_ms,
            )
            
        except Exception as e:
            logger.error("Schema extraction failed for {}: {}", url, str(e))
            return ScrapingResult(
                url=url,
                success=False,
                markdown=markdown_result.markdown,
                error=f"LLM extraction failed: {str(e)}",
                scrape_time_ms=markdown_result.scrape_time_ms,
            )
    
    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to clean markdown."""
        # Simple conversion - remove scripts, styles, and convert basic elements
        # For production, use a library like html2text or markdownify
        
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        
        # Convert headers
        for i in range(1, 7):
            html = re.sub(f'<h{i}[^>]*>(.*?)</h{i}>', f'\n{"#" * i} \\1\n', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert paragraphs and divs
        html = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<div[^>]*>(.*?)</div>', r'\n\1\n', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert links
        html = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert bold and italic
        html = re.sub(r'<(strong|b)[^>]*>(.*?)</\1>', r'**\2**', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<(em|i)[^>]*>(.*?)</\1>', r'*\2*', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert lists
        html = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert table cells (basic)
        html = re.sub(r'<td[^>]*>(.*?)</td>', r'| \1 ', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<th[^>]*>(.*?)</th>', r'| **\1** ', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<tr[^>]*>(.*?)</tr>', r'\1|\n', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove remaining HTML tags
        html = re.sub(r'<[^>]+>', '', html)
        
        # Clean up whitespace
        html = re.sub(r'\n\s*\n', '\n\n', html)
        html = re.sub(r' +', ' ', html)
        
        # Decode HTML entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')
        
        return html.strip()
    
    def _url_hash(self, url: str) -> str:
        """Generate hash for URL (for caching)."""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def _get_cached(self, url: str) -> Optional[ScrapingResult]:
        """Get cached result if available and not expired."""
        cache_file = self.cache_dir / f"{self._url_hash(url)}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > self.cache_ttl:
                return None  # Cache expired
            
            result = ScrapingResult(**data)
            result.cached = True
            return result
            
        except Exception as e:
            logger.warning("Cache read failed: {}", str(e))
            return None
    
    def _set_cached(self, url: str, result: ScrapingResult) -> None:
        """Cache scraping result."""
        cache_file = self.cache_dir / f"{self._url_hash(url)}.json"
        
        try:
            data = result.model_dump()
            data['timestamp'] = result.timestamp.isoformat()
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning("Cache write failed: {}", str(e))


# Convenience function for one-off scraping
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
