import asyncio
from playwright.async_api import async_playwright

async def dump_table():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://agmarknet.gov.in/daily-price-and-arrival-report", wait_until="networkidle")
        
        await asyncio.sleep(2)
        
        async def select_option(label_text, option_text):
            container = page.locator('div.relative.w-full').filter(has=page.locator(f'label:has-text("{label_text}")')).first
            peer = container.locator('.peer')
            await peer.click()
            await asyncio.sleep(1)
            option = page.locator(f'div[role="option"]:has-text("{option_text}"), li[role="option"]:has-text("{option_text}"), span.truncate:has-text("{option_text}"), div.truncate:has-text("{option_text}")').first
            await option.click(timeout=3000)
            await asyncio.sleep(1)

        await select_option("Price/Arrivals*", "Price")
        await select_option("Commodity Group*", "Vegetables")
        await select_option("Commodity*", "Tomato")
        await select_option("State*", "Karnataka")
        
        search_btn = page.locator('button:has-text("Go")').first
        await search_btn.click(timeout=2000)
            
        await page.wait_for_selector('table', timeout=10000)
        table_html = await page.locator('table').inner_html()

        with open('table_dump.html', 'w', encoding='utf-8') as f:
            f.write("<table>" + table_html + "</table>")
            
        print("Done dropping table!")

        await browser.close()

asyncio.run(dump_table())
