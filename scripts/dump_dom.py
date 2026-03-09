import asyncio
from playwright.async_api import async_playwright

async def dump_dom():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://agmarknet.gov.in/", wait_until="networkidle")
        
        # Give React some time to render
        await asyncio.sleep(3)
        
        content = await page.content()
        with open("agmarknet_rendered.html", "w", encoding="utf-8") as f:
            f.write(content)
        print("DOM dumped to agmarknet_rendered.html")
        await browser.close()

asyncio.run(dump_dom())
