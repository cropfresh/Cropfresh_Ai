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
from readability import Document
from markdownify import markdownify as md

# Local imports
from src.tools.browser_stealth import apply_stealth, get_random_user_agent, StealthConfig
from src.tools.web_search import WebSearchTool


class ScrapingResult(BaseModel):
    """Result from web scraping operation."""
    url: str
    success: bool
    markdown: str = ""
    html: str = ""
    extracted_data: Any = Field(default_factory=dict)
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
        fallback_query: Optional[str] = None,
    ) -> ScrapingResult:
        """
        Extract structured data using LLM and Pydantic schema.
        
        Args:
            url: URL to scrape
            schema: Pydantic model class for extraction
            instruction: Extraction instruction for LLM
            config: Scraping configuration
            fallback_query: Search query to use if the URL fails (e.g. 404/Timeout)
            
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
                instruction="Extract all commodity prices from the table",
                fallback_query="Agmarknet official portal live prices"
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
        
        # Implement Search Fallback if the URL fails
        if not markdown_result.success and fallback_query:
            logger.warning(f"URL {url} failed. Initiating search fallback for: {fallback_query}")
            search_tool = WebSearchTool()
            search_results = await search_tool.search(fallback_query, max_results=1)
            
            if search_results and search_results.results:
                new_url = search_results.results[0].url
                logger.info(f"Fallback Search found new URL: {new_url}. Retrying scrape...")
                url = new_url # Update URL for result reporting
                markdown_result = await self.scrape_to_markdown(new_url, config)
        
        if not markdown_result.success:
            return markdown_result
        
        try:
            extracted = await self._extract_chunks_with_llm(
                markdown=markdown_result.markdown,
                schema=schema,
                instruction=instruction
            )
            
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
            
    async def extract_with_schema(
        self,
        html_content: str,
        url: str,
        schema: Type[BaseModel],
        instruction: str,
    ) -> ScrapingResult:
        """
        Extract structured data using LLM and Pydantic schema from raw HTML.
        
        Args:
            html_content: Raw HTML content
            url: URL the content was fetched from (for tracking)
            schema: Pydantic model class for extraction
            instruction: Extraction instruction for LLM
            
        Returns:
            ScrapingResult with extracted data matching schema
        """
        if not self.llm_provider:
            return ScrapingResult(
                url=url,
                success=False,
                error="LLM provider not configured for schema extraction",
            )
            
        start_time = datetime.now()
        markdown = self._html_to_markdown(html_content)
        
        try:
            extracted = await self._extract_chunks_with_llm(
                markdown=markdown,
                schema=schema,
                instruction=instruction
            )
            
            return ScrapingResult(
                url=url,
                success=True,
                markdown=markdown,
                extracted_data=extracted,
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
            
        except Exception as e:
            logger.error("Schema extraction failed for {}: {}", url, str(e))
            return ScrapingResult(
                url=url,
                success=False,
                markdown=markdown,
                error=f"LLM extraction failed: {str(e)}",
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    async def _extract_chunks_with_llm(
        self,
        markdown: str,
        schema: Type[BaseModel],
        instruction: str,
        chunk_size: int = 15000,
    ) -> Any:
        """Helper to chunk markdown and extract data safely across all chunks."""
        chunks = []
        current_chunk = ""
        
        # Semantic chunking by paragraphs (double newlines)
        paragraphs = markdown.split("\n\n")
        
        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        if not chunks:
            return []
            
        all_results = []
        is_list_expected = False
        
        for chunk in chunks:
            extraction_prompt = f"""Extract data from the following web page content according to the schema.

INSTRUCTION: {instruction}

SCHEMA:
{json.dumps(schema.model_json_schema(), indent=2)}

PAGE CONTENT:
{chunk}

Return a valid JSON object or array matching the schema. Only output JSON, no explanation."""

            # Call LLM
            llm_response = await self.llm_provider.agenerate(extraction_prompt)
            
            # Parse JSON safely
            json_match = re.search(r'[\[\{].*[\]\}]', llm_response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                if isinstance(extracted, list):
                    is_list_expected = True
                    all_results.extend(extracted)
                elif isinstance(extracted, dict) and extracted:
                    # Ignore empty dict extrations from irrelevant chunks
                    all_results.append(extracted)
                    
        if is_list_expected:
            return all_results
            
        if all_results:
            return all_results[0]
            
        return {}
    
    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to clean markdown using Readability and Markdownify."""
        if not html or len(html.strip()) == 0:
            return ""

        try:
            # 1. Boilerplate Removal: Use Readability to extract the main content
            # This strips navbars, footers, sidebars, and ads
            doc = Document(html)
            main_html = doc.summary()
            
            # (Optional) We can also get the title if needed: title = doc.short_title()

            # 2. Convert to Markdown: Use markdownify on the clean HTML
            # We strip out images and links if we only care about text/tables, 
            # but for our schema extraction, tables are essential.
            markdown = md(
                main_html, 
                strip=['a', 'img', 'script', 'style'], 
                heading_style="ATX",
                bullets="-",
            )
            
            # Clean up excessive newlines
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)
            
            return markdown.strip()
            
        except Exception as e:
            logger.warning(f"Readability/Markdownify failed, falling back to basic extraction: {e}")
            # Fallback to simple regex if readability completely chokes on the DOM
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
            html = re.sub(r'<[^>]+>', ' ', html)
            html = re.sub(r'\s+', ' ', html)
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
