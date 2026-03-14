"""
Playwright-based navigator for Agmarknet React SPA.
Extracted to maintain <200 line rule.
"""
import asyncio
from typing import Any, Optional

from loguru import logger


class AgmarknetNavigator:
    """Handles the complex interaction with the Agmarknet 2.0 React SPA form."""

    @staticmethod
    async def fill_form(
        page: Any,
        commodity: str,
        state: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
    ) -> None:
        """
        Fills the primary search form on the React SPA using label targets.
        """
        logger.info("Filling React SPA form...")

        # Price/Arrivals* - Default to Prices
        await AgmarknetNavigator._select_dropdown(page, "Price/Arrivals*", "Price")

        # Commodity Group* — hardcoded to "Vegetables"
        await AgmarknetNavigator._select_dropdown(page, "Commodity Group*", "Vegetables")
        await asyncio.sleep(1.5)  # Wait for Commodity API to populate

        # Commodity*
        await AgmarknetNavigator._select_dropdown(page, "Commodity*", commodity)

        # State*
        if state:
            await AgmarknetNavigator._select_dropdown(page, "State*", state)
            await asyncio.sleep(1.0)

        # District
        if district:
            await AgmarknetNavigator._select_dropdown(page, "District", district)
            await asyncio.sleep(1.0)

        # Market
        if market:
            await AgmarknetNavigator._select_dropdown(page, "Market", market)

    @staticmethod
    async def _select_dropdown(
        page: Any, label_text: str, option_text: str, timeout: int = 4000
    ) -> None:
        """
        Clicks a custom generic dropdown based on its preceding label text,
        then selects the desired option.
        """
        if not option_text:
            return

        logger.debug(f"Selecting '{option_text}' in '{label_text}'...")

        try:
            # Find the container that has the exact string in a label
            container = page.locator('div.relative.w-full').filter(has=page.locator(f'label:has-text("{label_text}")')).first
            clickable = container.locator(".peer")

            await clickable.wait_for(state="visible", timeout=timeout)
            await asyncio.sleep(0.3)
            await clickable.click()
            await asyncio.sleep(0.8)

            # Options can be div, li, span with role="option" or truncate classes
            option_locator = page.locator(
                f'div[role="option"]:has-text("{option_text}"), '
                f'li[role="option"]:has-text("{option_text}"), '
                f'span.truncate:has-text("{option_text}"), '
                f'div.truncate:has-text("{option_text}")'
            ).first

            await option_locator.wait_for(state="visible", timeout=timeout)
            await option_locator.click()
            logger.debug(f"Successfully selected {option_text}")
        except Exception as e:
            logger.warning(f"Could not select '{option_text}' in '{label_text}': {e}")
            # Click again to close dropdown if failed
            try:
                await clickable.click()
            except Exception:
                pass
            raise ValueError(f"Option {option_text} not found for {label_text}")

        await asyncio.sleep(0.5)
