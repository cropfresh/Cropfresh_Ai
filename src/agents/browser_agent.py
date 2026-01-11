"""
BrowserAgent
============
Interactive browser automation for complex web tasks.

Use cases:
- Multi-step form submissions
- Authenticated scraping (login required)
- Dynamic content interaction
- Screenshot and visual validation

Author: CropFresh AI Team
Version: 1.0.0
"""

import asyncio
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

from src.tools.browser_stealth import (
    apply_stealth,
    apply_human_behavior,
    create_stealth_context,
    get_random_delay,
    get_random_user_agent,
)


class ActionType(str, Enum):
    """Browser action types."""
    GOTO = "goto"
    CLICK = "click"
    TYPE = "type"
    FILL = "fill"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    WAIT = "wait"
    WAIT_FOR_SELECTOR = "wait_for_selector"
    SELECT = "select"
    HOVER = "hover"
    PRESS = "press"
    GET_TEXT = "get_text"
    GET_ATTRIBUTE = "get_attribute"


class BrowserAction(BaseModel):
    """A single browser action."""
    action: ActionType
    selector: Optional[str] = None
    value: Optional[str] = None
    timeout: int = 30000
    options: dict = Field(default_factory=dict)


class BrowserResult(BaseModel):
    """Result from browser action."""
    success: bool
    action: ActionType
    selector: Optional[str] = None
    value: Optional[str] = None
    screenshot_path: Optional[str] = None
    page_content: Optional[str] = None
    extracted_text: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class BrowserSession(BaseModel):
    """Browser session info."""
    session_id: str
    start_time: datetime
    current_url: Optional[str] = None
    page_title: Optional[str] = None
    actions_count: int = 0


class BrowserAgent:
    """
    Agent for interactive browser automation.
    
    Designed for:
    - Navigating authenticated portals (eNAM login)
    - Multi-step data retrieval
    - Form submissions
    - Visual validation
    
    Usage:
        agent = BrowserAgent()
        await agent.start_session()
        
        # Navigate and interact
        await agent.execute_action(BrowserAction(action="goto", value="https://enam.gov.in"))
        await agent.execute_action(BrowserAction(action="click", selector="#login-btn"))
        await agent.execute_action(BrowserAction(action="fill", selector="#username", value="user@email.com"))
        
        # Get page content
        content = await agent.get_page_markdown()
        
        await agent.close_session()
    """
    
    def __init__(
        self,
        headless: bool = True,
        stealth: bool = True,
        screenshot_dir: Optional[Path] = None,
    ):
        """
        Initialize BrowserAgent.
        
        Args:
            headless: Run browser in headless mode
            stealth: Apply anti-detection measures
            screenshot_dir: Directory to save screenshots
        """
        self.headless = headless
        self.stealth = stealth
        self.screenshot_dir = screenshot_dir or Path("data/screenshots")
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._session: Optional[BrowserSession] = None
        
        # Ensure screenshot directory exists
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("BrowserAgent initialized (headless={}, stealth={})", headless, stealth)
    
    async def start_session(self, session_id: Optional[str] = None) -> BrowserSession:
        """
        Initialize browser and start a new session.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            BrowserSession with session info
        """
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=['--disable-blink-features=AutomationControlled'] if self.stealth else [],
        )
        
        if self.stealth:
            self._context = await create_stealth_context(self._browser)
        else:
            self._context = await self._browser.new_context()
        
        self._page = await self._context.new_page()
        
        # Apply stealth scripts
        if self.stealth:
            await apply_stealth(self._page)
        
        # Create session
        import uuid
        self._session = BrowserSession(
            session_id=session_id or str(uuid.uuid4())[:8],
            start_time=datetime.now(),
        )
        
        logger.info("Browser session started: {}", self._session.session_id)
        return self._session
    
    async def execute_action(self, action: BrowserAction) -> BrowserResult:
        """
        Execute a single browser action.
        
        Args:
            action: BrowserAction to execute
            
        Returns:
            BrowserResult with action outcome
        """
        if not self._page:
            return BrowserResult(
                success=False,
                action=action.action,
                error="No active session. Call start_session() first.",
            )
        
        start_time = datetime.now()
        
        try:
            result = await self._execute_action_internal(action)
            
            # Update session
            if self._session:
                self._session.actions_count += 1
                self._session.current_url = self._page.url
                self._session.page_title = await self._page.title()
            
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.debug("Action {} completed in {:.0f}ms", action.action, result.duration_ms)
            
            return result
            
        except Exception as e:
            logger.error("Action {} failed: {}", action.action, str(e))
            return BrowserResult(
                success=False,
                action=action.action,
                selector=action.selector,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
    
    async def _execute_action_internal(self, action: BrowserAction) -> BrowserResult:
        """Internal action execution with specific handlers."""
        
        match action.action:
            case ActionType.GOTO:
                await self._page.goto(
                    action.value,
                    timeout=action.timeout,
                    wait_until=action.options.get("wait_until", "networkidle"),
                )
                if self.stealth:
                    await apply_human_behavior(self._page)
                return BrowserResult(success=True, action=action.action, value=action.value)
            
            case ActionType.CLICK:
                await self._page.click(action.selector, timeout=action.timeout)
                await asyncio.sleep(get_random_delay(0.2, 0.5) if self.stealth else 0.1)
                return BrowserResult(success=True, action=action.action, selector=action.selector)
            
            case ActionType.TYPE:
                await self._page.type(
                    action.selector,
                    action.value,
                    delay=50 if self.stealth else 0,  # Human-like typing delay
                )
                return BrowserResult(success=True, action=action.action, selector=action.selector)
            
            case ActionType.FILL:
                await self._page.fill(action.selector, action.value, timeout=action.timeout)
                return BrowserResult(success=True, action=action.action, selector=action.selector)
            
            case ActionType.SCROLL:
                scroll_amount = int(action.value or 500)
                await self._page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                return BrowserResult(success=True, action=action.action, value=action.value)
            
            case ActionType.SCREENSHOT:
                filename = action.value or f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                path = self.screenshot_dir / filename
                await self._page.screenshot(
                    path=str(path),
                    full_page=action.options.get("full_page", True),
                )
                return BrowserResult(
                    success=True,
                    action=action.action,
                    screenshot_path=str(path),
                )
            
            case ActionType.WAIT:
                await asyncio.sleep(float(action.value or 1))
                return BrowserResult(success=True, action=action.action)
            
            case ActionType.WAIT_FOR_SELECTOR:
                await self._page.wait_for_selector(action.selector, timeout=action.timeout)
                return BrowserResult(success=True, action=action.action, selector=action.selector)
            
            case ActionType.SELECT:
                await self._page.select_option(action.selector, action.value)
                return BrowserResult(success=True, action=action.action, selector=action.selector)
            
            case ActionType.HOVER:
                await self._page.hover(action.selector, timeout=action.timeout)
                return BrowserResult(success=True, action=action.action, selector=action.selector)
            
            case ActionType.PRESS:
                await self._page.press(action.selector or "body", action.value)
                return BrowserResult(success=True, action=action.action, value=action.value)
            
            case ActionType.GET_TEXT:
                element = await self._page.query_selector(action.selector)
                text = await element.inner_text() if element else None
                return BrowserResult(
                    success=True,
                    action=action.action,
                    selector=action.selector,
                    extracted_text=text,
                )
            
            case ActionType.GET_ATTRIBUTE:
                element = await self._page.query_selector(action.selector)
                attr = await element.get_attribute(action.value) if element else None
                return BrowserResult(
                    success=True,
                    action=action.action,
                    selector=action.selector,
                    extracted_text=attr,
                )
            
            case _:
                return BrowserResult(
                    success=False,
                    action=action.action,
                    error=f"Unknown action type: {action.action}",
                )
    
    async def execute_workflow(self, actions: list[BrowserAction]) -> list[BrowserResult]:
        """
        Execute a sequence of browser actions.
        
        Args:
            actions: List of BrowserActions to execute in order
            
        Returns:
            List of BrowserResults
        """
        results = []
        
        for action in actions:
            result = await self.execute_action(action)
            results.append(result)
            
            # Stop on first failure if configured
            if not result.success:
                logger.warning("Workflow stopped due to action failure: {}", result.error)
                break
        
        return results
    
    async def get_page_content(self) -> Optional[str]:
        """Get current page HTML content."""
        if not self._page:
            return None
        return await self._page.content()
    
    async def get_page_markdown(self) -> Optional[str]:
        """Get current page content as markdown."""
        html = await self.get_page_content()
        if not html:
            return None
        
        # Use WebScrapingAgent's HTML to markdown conversion
        from src.agents.web_scraping_agent import WebScrapingAgent
        agent = WebScrapingAgent()
        return agent._html_to_markdown(html)
    
    async def get_page_info(self) -> dict:
        """Get current page information."""
        if not self._page:
            return {}
        
        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "viewport": self._page.viewport_size,
        }
    
    async def save_cookies(self, filepath: Path) -> None:
        """Save current session cookies to file."""
        if not self._context:
            return
        
        cookies = await self._context.cookies()
        import json
        with open(filepath, 'w') as f:
            json.dump(cookies, f, indent=2)
        
        logger.info("Cookies saved to {}", filepath)
    
    async def load_cookies(self, filepath: Path) -> None:
        """Load cookies from file into current session."""
        if not self._context or not filepath.exists():
            return
        
        import json
        with open(filepath, 'r') as f:
            cookies = json.load(f)
        
        await self._context.add_cookies(cookies)
        logger.info("Cookies loaded from {}", filepath)
    
    async def close_session(self) -> None:
        """Clean up browser resources."""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        if self._session:
            logger.info(
                "Session {} closed ({} actions, {:.1f}s)",
                self._session.session_id,
                self._session.actions_count,
                (datetime.now() - self._session.start_time).total_seconds(),
            )
        
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._session = None
    
    @property
    def session(self) -> Optional[BrowserSession]:
        """Get current session info."""
        return self._session
    
    @property
    def page(self) -> Optional[Page]:
        """Get current page for advanced operations."""
        return self._page
