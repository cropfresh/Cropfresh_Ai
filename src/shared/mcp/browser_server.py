"""
MCP Browser Server
==================
Model Context Protocol server for browser automation.

Exposes browser automation capabilities as MCP tools
that can be called by LLMs for web scraping and navigation.

Tools:
- navigate_to_url: Navigate and get page content
- scrape_structured_data: Extract data using LLM
- take_screenshot: Capture webpage screenshot
- get_mandi_prices: Agricultural market prices
- get_weather: Weather data for location

Author: CropFresh AI Team
Version: 1.0.0
"""

import asyncio
from typing import Optional, Any

from loguru import logger

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    logger.warning("MCP SDK not available")

from src.agents.web_scraping_agent import WebScrapingAgent, ScrapingConfig
from src.agents.browser_agent import BrowserAgent, BrowserAction, ActionType
from src.tools.agri_scrapers import AgriculturalDataAPI


# Initialize MCP server
if HAS_MCP:
    server = Server("cropfresh-browser")
else:
    server = None


# Global instances (lazy initialized)
_scraper: Optional[WebScrapingAgent] = None
_browser: Optional[BrowserAgent] = None
_agri_api: Optional[AgriculturalDataAPI] = None


async def get_scraper() -> WebScrapingAgent:
    """Get or create WebScrapingAgent instance."""
    global _scraper
    if _scraper is None:
        _scraper = WebScrapingAgent()
        await _scraper.initialize()
    return _scraper


async def get_browser() -> BrowserAgent:
    """Get or create BrowserAgent instance."""
    global _browser
    if _browser is None:
        _browser = BrowserAgent(headless=True, stealth=True)
        await _browser.start_session()
    return _browser


def get_agri_api() -> AgriculturalDataAPI:
    """Get or create AgriculturalDataAPI instance."""
    global _agri_api
    if _agri_api is None:
        _agri_api = AgriculturalDataAPI()
    return _agri_api


# ============================================================================
# MCP Tool Definitions
# ============================================================================

if HAS_MCP:
    
    @server.tool()
    async def navigate_to_url(url: str) -> str:
        """
        Navigate browser to a URL and return page content as markdown.
        
        Args:
            url: The URL to navigate to
            
        Returns:
            Page content in markdown format
        """
        try:
            scraper = await get_scraper()
            result = await scraper.scrape_to_markdown(url)
            
            if result.success:
                return result.markdown
            else:
                return f"Error: {result.error}"
                
        except Exception as e:
            logger.error("navigate_to_url failed: {}", str(e))
            return f"Error: {str(e)}"
    
    
    @server.tool()
    async def scrape_structured_data(
        url: str,
        data_description: str,
        fields: list[str],
    ) -> dict:
        """
        Scrape structured data from a URL using LLM extraction.
        
        Args:
            url: The URL to scrape
            data_description: Description of the data to extract
            fields: List of field names to extract
            
        Returns:
            Dictionary with extracted data
        """
        try:
            scraper = await get_scraper()
            
            # Build CSS selectors from field names (basic heuristic)
            selectors = {}
            for field in fields:
                # Try common patterns
                selectors[field] = f"[data-{field}], .{field}, #{field}"
            
            result = await scraper.scrape_with_css(url, selectors)
            
            if result.success:
                return result.extracted_data
            else:
                return {"error": result.error}
                
        except Exception as e:
            logger.error("scrape_structured_data failed: {}", str(e))
            return {"error": str(e)}
    
    
    @server.tool()
    async def take_screenshot(url: str, filename: Optional[str] = None) -> str:
        """
        Take a screenshot of a webpage.
        
        Args:
            url: The URL to screenshot
            filename: Optional filename for the screenshot
            
        Returns:
            Path to the saved screenshot
        """
        try:
            browser = await get_browser()
            
            await browser.execute_action(BrowserAction(
                action=ActionType.GOTO,
                value=url,
            ))
            
            result = await browser.execute_action(BrowserAction(
                action=ActionType.SCREENSHOT,
                value=filename,
            ))
            
            if result.success:
                return result.screenshot_path or "Screenshot taken"
            else:
                return f"Error: {result.error}"
                
        except Exception as e:
            logger.error("take_screenshot failed: {}", str(e))
            return f"Error: {str(e)}"
    
    
    @server.tool()
    async def get_mandi_prices(
        commodity: str,
        state: Optional[str] = None,
    ) -> list[dict]:
        """
        Get current mandi prices for a commodity.
        
        Args:
            commodity: Name of the commodity (e.g., "Tomato", "Onion")
            state: Optional state filter
            
        Returns:
            List of price records with mandi, price, and date
        """
        try:
            api = get_agri_api()
            prices = await api.get_mandi_prices(commodity, state)
            
            return [p.model_dump() for p in prices]
            
        except Exception as e:
            logger.error("get_mandi_prices failed: {}", str(e))
            return [{"error": str(e)}]
    
    
    @server.tool()
    async def get_weather(
        state: str,
        district: Optional[str] = None,
    ) -> list[dict]:
        """
        Get weather forecast for a location.
        
        Args:
            state: State name
            district: Optional district name
            
        Returns:
            Weather forecast data
        """
        try:
            api = get_agri_api()
            weather = await api.get_weather(state, district)
            
            return [w.model_dump() for w in weather]
            
        except Exception as e:
            logger.error("get_weather failed: {}", str(e))
            return [{"error": str(e)}]
    
    
    @server.tool()
    async def get_agri_news(
        source: str = "rural_voice",
        limit: int = 5,
    ) -> list[dict]:
        """
        Get agricultural news articles.
        
        Args:
            source: News source ("rural_voice" or "agri_farming")
            limit: Maximum number of articles
            
        Returns:
            List of news articles
        """
        try:
            api = get_agri_api()
            news = await api.get_news(source, limit)
            
            return [n.model_dump() for n in news]
            
        except Exception as e:
            logger.error("get_agri_news failed: {}", str(e))
            return [{"error": str(e)}]


# ============================================================================
# Server Lifecycle
# ============================================================================

async def cleanup():
    """Clean up resources on shutdown."""
    global _scraper, _browser, _agri_api
    
    if _scraper:
        await _scraper.close()
        _scraper = None
    
    if _browser:
        await _browser.close_session()
        _browser = None
    
    if _agri_api:
        await _agri_api.close_all()
        _agri_api = None
    
    logger.info("MCP Browser Server resources cleaned up")


def run_server():
    """Run the MCP server."""
    if not HAS_MCP:
        logger.error("MCP SDK not installed. Run: pip install mcp")
        return
    
    import signal
    
    loop = asyncio.get_event_loop()
    
    # Handle shutdown signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(cleanup()))
    
    try:
        logger.info("Starting CropFresh MCP Browser Server...")
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    finally:
        loop.run_until_complete(cleanup())


if __name__ == "__main__":
    run_server()
