import asyncio
from playwright.async_api import async_playwright

async def debug_dropdowns():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://agmarknet.gov.in/daily-price-and-arrival-report", wait_until="networkidle")
        
        await asyncio.sleep(2)
        
        async def select_option(label_text, option_text):
            print(f"Selecting {option_text} in {label_text}")
            # Find the container that has a label with exact text label_text
            container = page.locator('div.relative.w-full').filter(has=page.locator(f'label:has-text("{label_text}")')).first
            
            # Click the peer div
            peer = container.locator('.peer')
            await peer.click()
            await asyncio.sleep(1)
            
            # Now find the option that appeared
            option = page.locator(f'div[role="option"]:has-text("{option_text}"), li[role="option"]:has-text("{option_text}"), span.truncate:has-text("{option_text}"), div.truncate:has-text("{option_text}")').first
            try:
                await option.click(timeout=3000)
                print(f"Successfully clicked {option_text}")
            except Exception as e:
                print(f"Option not found! Error: {e}")
                # Dump options text for debugging
                opts = await page.locator('[role="option"], .truncate').all()
                texts = [await o.text_content() for o in opts]
                print(f"Available options are: {texts[:20]}")
            await asyncio.sleep(1)

        await select_option("Price/Arrivals*", "Prices")
        await select_option("Commodity Group*", "Vegetables")
        
        # Try to select Beans, which might fail
        try:
            await select_option("Commodity*", "Beans")
        except Exception as e:
            pass # The select_option function already dumps available options if it fails
            
        await select_option("State*", "Karnataka")
        
        print("Clicking Search...")
        try:
            search_btn = page.locator('button:has-text("Show Data")').first
            await search_btn.click(timeout=2000)
            print("Clicked 'Show Data'")
        except:
            search_btn = page.locator('button:has-text("Go")').first
            await search_btn.click(timeout=2000)
            print("Clicked 'Go'")
            
        try:
            # Wait a few seconds for any API response
            await asyncio.sleep(5)
            # Take a screenshot to see what's on the screen
            await page.screenshot(path='beans_error.png', full_page=True)
            print("Saved screenshot to beans_error.png")
            
            await page.wait_for_selector('table', timeout=5000)
            table_html = await page.locator('table').inner_html()
            print(f"Table content prefix: {table_html[:200]}")
        except Exception as e:
            print("Table not found. Dumping body HTML instead...")
            body_html = await page.locator('body').inner_html()
            with open('beans_error.html', 'w', encoding='utf-8') as f:
                f.write(body_html)

        await browser.close()

asyncio.run(debug_dropdowns())
