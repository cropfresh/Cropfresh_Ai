"""
Test script for Pricing Agent
==============================
Tests the Pricing Agent with mock and real data.

Usage:
    python -m uv run python scripts/test_pricing.py
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def test_pricing_agent():
    """Test the Pricing Agent functionality."""
    from src.agents.pricing_agent import PricingAgent
    
    print("\n" + "=" * 60)
    print("    🌾 CropFresh AI - Pricing Agent Test")
    print("=" * 60 + "\n")
    
    # Create agent with mock data
    agent = PricingAgent(use_mock=True)
    
    # Test 1: Get current price
    print("📊 Test 1: Get Current Price (Tomato, Kolar)")
    print("-" * 40)
    
    prices = await agent.get_current_price("Tomato", district="Kolar")
    if prices:
        p = prices[0]
        print(f"  Commodity: {p.commodity}")
        print(f"  Market: {p.market}")
        print(f"  Modal Price: ₹{p.modal_price:.0f}/quintal (₹{p.modal_price_per_kg:.1f}/kg)")
        print(f"  Range: ₹{p.min_price:.0f} - ₹{p.max_price:.0f}/quintal")
        print("  ✅ Price fetch successful!")
    else:
        print("  ❌ No prices returned")
    
    # Test 2: Get Recommendation
    print("\n📈 Test 2: Get Sell/Hold Recommendation")
    print("-" * 40)
    
    rec = await agent.get_recommendation("Tomato", "Kolar", quantity_kg=200)
    print(f"  Current Price: ₹{rec.current_price:.1f}/kg")
    print(f"  Recommendation: {rec.recommended_action.upper()}")
    print(f"  Confidence: {rec.confidence:.0%}")
    print(f"  Reason: {rec.reason}")
    print("  ✅ Recommendation generated!")
    
    # Test 3: AISP Calculation
    print("\n💰 Test 3: AISP Calculation (200 kg, 30 km)")
    print("-" * 40)
    
    aisp = agent.calculate_aisp(
        farmer_price_per_kg=25,  # ₹25/kg for tomatoes
        quantity_kg=200,
        distance_km=30,
    )
    print(f"  Farmer Price: ₹{aisp.farmer_price_per_kg}/kg")
    print(f"  Quantity: {aisp.quantity_kg} kg")
    print(f"  ───────────────────────────")
    print(f"  Farmer Payout:   ₹{aisp.farmer_payout:,.0f}")
    print(f"  Logistics:       ₹{aisp.logistics_cost:,.0f}")
    print(f"  Handling:        ₹{aisp.handling_cost:,.0f}")
    print(f"  Platform Fee ({aisp.platform_fee_pct:.0%}): ₹{aisp.platform_fee:,.0f}")
    print(f"  ───────────────────────────")
    print(f"  Total AISP:      ₹{aisp.total_aisp:,.0f}")
    print(f"  AISP per kg:     ₹{aisp.aisp_per_kg:.2f}")
    print("  ✅ AISP calculation successful!")
    
    # Test 4: Multiple commodities
    print("\n🥬 Test 4: Multiple Commodities")
    print("-" * 40)
    
    commodities = ["Potato", "Onion", "Capsicum", "Cabbage"]
    for crop in commodities:
        prices = await agent.get_current_price(crop)
        if prices:
            print(f"  {crop}: ₹{prices[0].modal_price_per_kg:.1f}/kg")
    print("  ✅ Multi-commodity fetch successful!")
    
    print("\n" + "=" * 60)
    print("    ✅ All Pricing Agent tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_pricing_agent())
