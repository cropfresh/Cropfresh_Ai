import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.scrapers.agmarknet_api import AgmarknetTool

async def test_api():
    tool = AgmarknetTool()
    print("Testing CEDA fallback API...")
    prices = await tool._fallback_ceda(commodity="Tomato", state="Karnataka", district="Kolar", limit=5)
    print(f"Got {len(prices)} prices from CEDA:")
    for p in prices:
        print(p)

    print("\nTesting get_mock_prices...")
    mocks = tool.get_mock_prices(commodity="Tomato")
    print(f"Got mock: {mocks[0]}")

asyncio.run(test_api())
