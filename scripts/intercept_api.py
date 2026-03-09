import asyncio
import json
from playwright.async_api import async_playwright
from loguru import logger

async def intercept_and_search():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Add event listeners for network requests
        context = await browser.new_context()
        page = await context.new_page()
        
        # Store intercepted requests
        api_requests = []
        
        async def handle_request(request):
            # Capture all XHR/fetch requests, ignore images/css/scripts
            if request.resource_type in ["fetch", "xhr"]:
                try:
                    api_requests.append({
                        "url": request.url,
                        "method": request.method,
                        "headers": request.headers,
                        "post_data": request.post_data if request.post_data else None
                    })
                    logger.info(f"Intercepted {request.method} request to: {request.url}")
                except Exception as e:
                    logger.debug(f"Could not parse request: {e}")
                    
        page.on("request", handle_request)
        
        logger.info("Navigating to Agmarknet...")
        await page.goto("https://agmarknet.gov.in/daily-price-and-arrival-report", wait_until="networkidle")
        await asyncio.sleep(2)
        
        async def select_option(label_text, option_text):
            logger.info(f"Selecting {option_text} in {label_text}")
            try:
                main_container = page.locator('div.relative.flex.flex-col').filter(has=page.locator(f'label:has-text("{label_text}")')).first
                await main_container.locator('.peer').click(timeout=3000)
                await asyncio.sleep(0.5)
                # Find exact match with span.truncate
                await main_container.locator(f'span.truncate:text-is("{option_text}")').first.click(timeout=3000)
                await asyncio.sleep(1) # Wait for network
            except Exception as e:
                logger.error(f"Failed to select {option_text}: {e}")
                
        # Fill out the mandatory form fields
        await select_option("Price/Arrivals*", "Price")
        await select_option("Commodity Group*", "Vegetables")
        await select_option("Commodity*", "Cluster beans")
        await select_option("State*", "Karnataka")
        
        logger.info("Clicking Search...")
        try:
            search_btn = page.locator('button:has-text("Show Data")').first
            await search_btn.click(timeout=2000)
        except:
            search_btn = page.locator('button:has-text("Go")').first
            await search_btn.click(timeout=2000)
            
        # Wait a bit for the API call to complete
        logger.info("Waiting for data response...")
        await asyncio.sleep(5)
        
        # Dump the intercepted requests to a file
        with open('intercepted_requests.json', 'w', encoding='utf-8') as f:
            json.dump(api_requests, f, indent=4)
            
        logger.info(f"Successfully intercepted {len(api_requests)} POST requests. Saved to intercepted_requests.json")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(intercept_and_search())
