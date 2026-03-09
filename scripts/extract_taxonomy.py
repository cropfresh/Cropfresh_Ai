import asyncio
import json
from playwright.async_api import async_playwright
from loguru import logger

async def extract_options(page, label_text: str):
    """Opens a dropdown and extracts all its options."""
    logger.info(f"Extracting options for: {label_text}")
    try:
        # Find the main container holding both the input and the dropdown popup
        main_container = page.locator('div.relative.flex.flex-col').filter(has=page.locator(f'label:has-text("{label_text}")')).first
        peer = main_container.locator('.peer')
        
        # Click to open the dropdown
        await peer.click(timeout=3000)
        await asyncio.sleep(1) # Wait for animation/render
        
        # Get all options text specifically from span.truncate inside this container
        options = await main_container.locator('span.truncate').all_text_contents()
        
        # Clean up options (remove empty ones, strip whitespace)
        options = [opt.strip() for opt in options if opt.strip()]
        
        # Click the label or body to close the dropdown properly without selecting anything
        await page.mouse.click(0, 0)
        await asyncio.sleep(0.5)
        
        return options
    except Exception as e:
        logger.error(f"Failed to extract {label_text}: {e}")
        # Try to click away to close any stuck dropdown
        await page.mouse.click(0, 0)
        return []

async def extract_taxonomy():
    taxonomy = {
        "price_arrivals": [],
        "commodity_groups": {},
        "states": {}
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        logger.info("Navigating to Agmarknet...")
        await page.goto("https://agmarknet.gov.in/daily-price-and-arrival-report", wait_until="networkidle")
        await asyncio.sleep(3)
        
        # 1. Price/Arrival Options
        taxonomy["price_arrivals"] = await extract_options(page, "Price/Arrivals*")
        
        # 2. Extract Commodity Hierarchy
        logger.info("Extracting Commodity Groups...")
        commodity_groups = await extract_options(page, "Commodity Group*")
        
        for group in commodity_groups:
            logger.info(f"Processing group: {group}")
            taxonomy["commodity_groups"][group] = []
            
            # Select the group
            main_container = page.locator('div.relative.flex.flex-col').filter(has=page.locator('label:has-text("Commodity Group*")')).first
            await main_container.locator('.peer').click()
            await asyncio.sleep(0.5)
            await main_container.locator(f'span.truncate:text-is("{group}")').first.click()
            
            # Wait for Commodity dropdown to populate (API call)
            await asyncio.sleep(2)
            
            options = await extract_options(page, "Commodity*")
            taxonomy["commodity_groups"][group] = options

        # 3. Extract State and District Hierarchy
        logger.info("Extracting States...")
        states = await extract_options(page, "State*")
        
        for state in states:
            logger.info(f"Processing state: {state}")
            taxonomy["states"][state] = []
            
            # Select the state
            main_container = page.locator('div.relative.flex.flex-col').filter(has=page.locator('label:has-text("State*")')).first
            await main_container.locator('.peer').click()
            await asyncio.sleep(0.5)
            await main_container.locator(f'span.truncate:text-is("{state}")').first.click()
            
            # Wait for District dropdown to populate (API call)
            await asyncio.sleep(2)
            
            options = await extract_options(page, "District")
            taxonomy["states"][state] = options
            
        # Write to JSON
        with open('agmarknet_taxonomy.json', 'w', encoding='utf-8') as f:
            json.dump(taxonomy, f, indent=4, ensure_ascii=False)
            
        logger.info("Taxonomy extraction complete! Saved to agmarknet_taxonomy.json")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(extract_taxonomy())
