import asyncio
from scrapling.fetchers import StealthyFetcher

async def test_enam():
    url = "https://enam.gov.in/web/dashboard/trade-data"
    print(f"Fetching {url}...")
    try:
        page = await StealthyFetcher.async_fetch(url)
        print("Success! Title:", page.title)
        rows = page.css("table.table tbody tr")
        print(f"Found {len(rows)} rows.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_enam())
