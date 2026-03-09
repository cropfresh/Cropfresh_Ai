import asyncio
from playwright.async_api import async_playwright

async def find_api():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Intercept network requests
        urls_seen = []
        page.on("response", lambda response: urls_seen.append((response.request.method, response.url, response.status)))
        
        url = "https://agmarknet.gov.in"
        print(f"Going to {url}...")
        try:
            await page.goto(url, wait_until="networkidle")
            await asyncio.sleep(2) # just in case
        except Exception as e:
            print(f"Error navigating: {e}")
        
        print("\n--- Network Responses ---")
        for method, u, status in urls_seen:
            if "api" in u.lower() or "json" in u.lower() or "graphql" in u.lower() or "gov.in" in u:
                if not u.endswith(".js") and not u.endswith(".css") and not u.endswith(".png") and not u.endswith(".woff2"):
                    print(f"{method} {status} {u}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(find_api())
