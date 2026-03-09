import asyncio
from playwright.async_api import async_playwright

async def debug_dropdowns():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://agmarknet.gov.in/daily-price-and-arrival-report", wait_until="networkidle")
        
        await asyncio.sleep(3)
        
        # Print all visible span.truncate elements
        elements = await page.locator("span.truncate").all()
        print(f"Found {len(elements)} span.truncate elements")
        
        for i, el in enumerate(elements[:50]):
            text = await el.text_content()
            print(f"{i}: {text}")
            
        print("\nNow clicking Commodity dropdown (index 4)...")
        container = page.locator("div.relative.w-full").nth(4)
        clickable = container.locator(".peer")
        await clickable.click()
        await asyncio.sleep(2)
        
        content = await page.content()
        with open("agmarknet_dropdown.html", "w", encoding="utf-8") as f:
            f.write(content)
        print("Wrote dropdown HTML to agmarknet_dropdown.html")

        await browser.close()

asyncio.run(debug_dropdowns())
