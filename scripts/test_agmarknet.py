"""
Test script for Agmarknet API
===============================
Tests the real Agmarknet API connection with actual market data.

Usage:
    python -m uv run python scripts/test_agmarknet.py
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()


async def test_agmarknet_api():
    """Test the Agmarknet API with real data."""
    from src.tools.agmarknet import AgmarknetTool
    
    print("\n" + "=" * 60)
    print("    🌾 CropFresh AI - Agmarknet API Test")
    print("=" * 60 + "\n")
    
    api_key = os.getenv("AGMARKNET_API_KEY", "")
    
    if not api_key or api_key == "your_data_gov_in_api_key":
        print("❌ AGMARKNET_API_KEY not set in .env")
        print("   Using mock data instead...")
        tool = AgmarknetTool(api_key="")
        prices = tool.get_mock_prices("Tomato", "Karnataka", "Kolar")
        print(f"\n📊 Mock Price for Tomato (Kolar):")
        if prices:
            p = prices[0]
            print(f"   Modal Price: ₹{p.modal_price:.0f}/quintal (₹{p.modal_price_per_kg:.1f}/kg)")
        return
    
    print(f"✅ API Key detected (length: {len(api_key)} chars)")
    
    # Create tool with real API key
    tool = AgmarknetTool(api_key=api_key)
    
    # Test 1: Fetch Tomato prices
    print("\n📊 Test 1: Fetching Tomato prices from Karnataka")
    print("-" * 40)
    
    try:
        prices = await tool.get_prices("Tomato", "Karnataka", limit=5)
        
        if prices:
            print(f"   ✅ Found {len(prices)} price records!\n")
            for i, p in enumerate(prices[:5], 1):
                print(f"   {i}. {p.market}")
                print(f"      Date: {p.date.strftime('%d %b %Y')}")
                print(f"      Modal: ₹{p.modal_price:.0f}/quintal (₹{p.modal_price_per_kg:.1f}/kg)")
                print(f"      Range: ₹{p.min_price:.0f} - ₹{p.max_price:.0f}/quintal")
                print()
        else:
            print("   ⚠️ No prices returned - API may be down or query empty")
            print("   Trying CEDA fallback...")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Fetch prices for multiple commodities
    print("\n🥬 Test 2: Multiple Commodities (Karnataka)")
    print("-" * 40)
    
    commodities = ["Onion", "Potato", "Capsicum"]
    
    for commodity in commodities:
        try:
            prices = await tool.get_prices(commodity, "Karnataka", limit=1)
            if prices:
                p = prices[0]
                print(f"   {commodity}: ₹{p.modal_price_per_kg:.1f}/kg ({p.market})")
            else:
                print(f"   {commodity}: No data")
        except Exception as e:
            print(f"   {commodity}: Error - {str(e)[:30]}")
    
    # Test 3: Specific district
    print("\n📍 Test 3: Kolar District Prices")
    print("-" * 40)
    
    try:
        prices = await tool.get_prices("Tomato", "Karnataka", district="Kolar", limit=3)
        if prices:
            for p in prices:
                print(f"   {p.market}: ₹{p.modal_price:.0f}/quintal")
        else:
            print("   No specific Kolar data found")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("    ✅ Agmarknet API test complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_agmarknet_api())
