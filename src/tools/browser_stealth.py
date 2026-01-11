"""
Browser Stealth Utilities
=========================
Anti-detection measures for web scraping.

Features:
- User-agent rotation
- Viewport randomization
- WebDriver property hiding
- Plugin fingerprint randomization
- Human-like delays

Author: CropFresh AI Team
Version: 1.0.0
"""

import random
from typing import Optional

from loguru import logger
from playwright.async_api import Page

try:
    from fake_useragent import UserAgent
    _ua = UserAgent()
    HAS_FAKE_UA = True
except ImportError:
    HAS_FAKE_UA = False
    logger.warning("fake_useragent not installed, using fallback user agents")


class StealthConfig:
    """Configuration for stealth browsing."""
    
    # Fallback user agents if fake_useragent not available
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]
    
    # Viewport sizes to rotate (common screen resolutions)
    VIEWPORTS = [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1536, "height": 864},
        {"width": 1440, "height": 900},
        {"width": 1280, "height": 720},
        {"width": 1600, "height": 900},
    ]
    
    # Common screen color depths
    COLOR_DEPTHS = [24, 32]
    
    # Common timezone offsets (IST for India)
    TIMEZONE_ID = "Asia/Kolkata"
    
    # Languages
    LANGUAGES = ["en-IN", "hi-IN", "en-US", "en-GB"]


def get_random_user_agent() -> str:
    """Get a random realistic user agent."""
    if HAS_FAKE_UA:
        try:
            return _ua.random
        except Exception:
            pass
    return random.choice(StealthConfig.USER_AGENTS)


def get_random_viewport() -> dict:
    """Get a random viewport size."""
    return random.choice(StealthConfig.VIEWPORTS)


def get_random_delay(min_seconds: float = 0.5, max_seconds: float = 2.0) -> float:
    """Get random delay between actions (human-like)."""
    return random.uniform(min_seconds, max_seconds)


async def apply_stealth(page: Page) -> None:
    """
    Apply anti-detection scripts to page.
    
    This helps avoid bot detection by:
    - Hiding webdriver property
    - Randomizing plugins/languages
    - Overriding navigator properties
    """
    
    # Hide webdriver property
    await page.add_init_script("""
        // Hide webdriver
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
        
        // Override plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => {
                return [
                    { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                    { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                    { name: 'Native Client', filename: 'internal-nacl-plugin' },
                ];
            }
        });
        
        // Override languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-IN', 'en-US', 'en', 'hi']
        });
        
        // Hide automation indicators
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
        
        // Override permissions query
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        
        // Add chrome object for Chrome detection
        window.chrome = {
            runtime: {},
            loadTimes: function() {},
            csi: function() {},
            app: {}
        };
        
        // Override connection type
        Object.defineProperty(navigator, 'connection', {
            get: () => ({
                effectiveType: '4g',
                rtt: 50,
                downlink: 10,
                saveData: false
            })
        });
    """)
    
    logger.debug("Stealth scripts applied to page")


async def apply_human_behavior(page: Page) -> None:
    """
    Simulate human-like behavior on the page.
    
    Actions:
    - Random mouse movements
    - Random scroll
    - Random delays
    """
    import asyncio
    
    # Random initial delay
    await asyncio.sleep(get_random_delay(0.3, 1.0))
    
    # Random mouse movement
    viewport = page.viewport_size
    if viewport:
        x = random.randint(100, viewport['width'] - 100)
        y = random.randint(100, min(400, viewport['height'] - 100))
        await page.mouse.move(x, y)
    
    # Small delay
    await asyncio.sleep(get_random_delay(0.1, 0.3))
    
    # Random scroll
    scroll_amount = random.randint(100, 500)
    await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
    
    logger.debug("Human behavior simulation applied")


async def create_stealth_context(
    browser,
    user_agent: Optional[str] = None,
    viewport: Optional[dict] = None,
) -> "BrowserContext":
    """
    Create a browser context with stealth settings.
    
    Args:
        browser: Playwright browser instance
        user_agent: Custom user agent (random if not provided)
        viewport: Custom viewport (random if not provided)
        
    Returns:
        BrowserContext with stealth configuration
    """
    ua = user_agent or get_random_user_agent()
    vp = viewport or get_random_viewport()
    
    context = await browser.new_context(
        user_agent=ua,
        viewport=vp,
        locale=random.choice(StealthConfig.LANGUAGES),
        timezone_id=StealthConfig.TIMEZONE_ID,
        color_scheme="light",
        has_touch=False,
        is_mobile=False,
        java_script_enabled=True,
        bypass_csp=True,  # Bypass Content Security Policy
        ignore_https_errors=True,
    )
    
    logger.debug("Stealth context created (UA: {}...)", ua[:50])
    return context


class RequestInterceptor:
    """
    Intercept and modify requests to avoid detection.
    """
    
    # Headers to always include
    DEFAULT_HEADERS = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }
    
    # URLs to block (tracking, analytics)
    BLOCKED_PATTERNS = [
        "*google-analytics.com*",
        "*googletagmanager.com*",
        "*facebook.com/tr*",
        "*doubleclick.net*",
        "*hotjar.com*",
        "*clarity.ms*",
    ]
    
    @classmethod
    async def setup(cls, page: Page) -> None:
        """Setup request interception on a page."""
        
        async def handle_route(route):
            url = route.request.url
            
            # Block tracking requests
            for pattern in cls.BLOCKED_PATTERNS:
                pattern_regex = pattern.replace("*", ".*")
                import re
                if re.match(pattern_regex, url):
                    await route.abort()
                    return
            
            # Add headers to requests
            headers = {**route.request.headers, **cls.DEFAULT_HEADERS}
            await route.continue_(headers=headers)
        
        await page.route("**/*", handle_route)
        logger.debug("Request interceptor setup complete")
