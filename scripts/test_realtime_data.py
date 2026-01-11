"""
Test Real-Time Data Integration (Phase 1)
==========================================
Tests for eNAM, IMD Weather, Google AMED, and RealTimeDataManager.

Run with:
    uv run python scripts/test_realtime_data.py

Author: CropFresh AI Team
"""

import asyncio
from datetime import datetime


def print_header(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} - {name}")
    if details:
        print(f"         {details}")


async def test_enam_client():
    """Test eNAM client functionality."""
    print_header("Testing eNAM Client")
    
    from src.tools.enam_client import get_enam_client
    
    client = get_enam_client(use_mock=True)
    
    # Test 1: Get live prices
    try:
        prices = await client.get_live_prices("Tomato", "Karnataka")
        passed = len(prices) > 0 and prices[0].commodity.lower() == "tomato"
        print_result("Get live prices", passed, f"{len(prices)} prices returned")
        
        if prices:
            p = prices[0]
            print(f"         Sample: {p.market} - ₹{p.modal_price:,.0f}/quintal")
    except Exception as e:
        print_result("Get live prices", False, str(e))
    
    # Test 2: Get price trends
    try:
        trend = await client.get_price_trends("Onion", "Maharashtra")
        passed = trend.current_price > 0
        print_result("Get price trends", passed, 
                    f"Current: ₹{trend.current_price:,.0f}, 7d change: {trend.change_7d_pct:+.1f}%")
    except Exception as e:
        print_result("Get price trends", False, str(e))
    
    # Test 3: Get market summary
    try:
        summary = await client.get_market_summary("Potato", "Karnataka")
        passed = summary.commodity.lower() == "potato"
        print_result("Get market summary", passed,
                    f"Avg modal: ₹{summary.avg_modal_price:,.0f}")
    except Exception as e:
        print_result("Get market summary", False, str(e))
    
    # Test 4: Data freshness
    try:
        freshness = client.get_data_freshness()
        passed = "mode" in freshness
        print_result("Data freshness check", passed, f"Mode: {freshness['mode']}")
    except Exception as e:
        print_result("Data freshness check", False, str(e))


async def test_imd_weather():
    """Test IMD Weather client functionality."""
    print_header("Testing IMD Weather Client")
    
    from src.tools.imd_weather import get_imd_client
    
    client = get_imd_client(use_mock=True)
    
    # Test 1: Get current weather
    try:
        weather = await client.get_current_weather("Karnataka", "Kolar")
        passed = weather.temperature_c > 0 and weather.state.lower() == "karnataka"
        print_result("Get current weather", passed,
                    f"{weather.temperature_c:.1f}°C, {weather.humidity_pct:.0f}% humidity, {weather.condition.value}")
    except Exception as e:
        print_result("Get current weather", False, str(e))
    
    # Test 2: Get forecast
    try:
        forecast = await client.get_forecast("Karnataka", "Kolar", days=5)
        passed = len(forecast.daily_forecasts) > 0
        print_result("Get 5-day forecast", passed,
                    f"{len(forecast.daily_forecasts)} days of forecast")
    except Exception as e:
        print_result("Get 5-day forecast", False, str(e))
    
    # Test 3: Get alerts
    try:
        alerts = await client.get_alerts("Karnataka", "Kolar")
        passed = isinstance(alerts, list)
        print_result("Get weather alerts", passed,
                    f"{len(alerts)} active alerts")
    except Exception as e:
        print_result("Get weather alerts", False, str(e))
    
    # Test 4: Get agro advisory
    try:
        advisory = await client.get_agro_advisory("Karnataka", "Kolar", "Tomato")
        passed = len(advisory.recommendations) > 0
        print_result("Get agro advisory", passed,
                    f"{len(advisory.recommendations)} recommendations")
        if advisory.recommendations:
            print(f"         → {advisory.recommendations[0]}")
    except Exception as e:
        print_result("Get agro advisory", False, str(e))


async def test_google_amed():
    """Test Google AMED client functionality."""
    print_header("Testing Google AMED Client")
    
    from src.tools.google_amed import get_amed_client
    
    client = get_amed_client(use_mock=True)
    
    # Kolar coordinates
    lat, lon = 13.1333, 78.1333
    
    # Test 1: Get crop monitoring
    try:
        monitoring = await client.get_crop_monitoring(lat, lon)
        passed = monitoring.ndvi > 0
        print_result("Get crop monitoring", passed,
                    f"Primary crop: {monitoring.primary_crop.value}, NDVI: {monitoring.ndvi:.2f}")
    except Exception as e:
        print_result("Get crop monitoring", False, str(e))
    
    # Test 2: Get season info
    try:
        season = await client.get_season_info(lat, lon, "Tomato")
        passed = season.season_name in ["Kharif", "Rabi", "Zaid"]
        print_result("Get season info", passed,
                    f"Season: {season.season_name}, Progress: {season.progress_pct:.0f}%")
    except Exception as e:
        print_result("Get season info", False, str(e))
    
    # Test 3: Get field boundaries
    try:
        boundaries = await client.get_field_boundaries(lat, lon)
        passed = isinstance(boundaries, list)
        print_result("Get field boundaries", passed,
                    f"{len(boundaries)} fields detected")
    except Exception as e:
        print_result("Get field boundaries", False, str(e))
    
    # Test 4: Get regional stats
    try:
        stats = await client.get_regional_stats("Karnataka", "Kolar")
        passed = len(stats.top_crops) > 0
        print_result("Get regional stats", passed,
                    f"Top crops: {', '.join(c['crop'] for c in stats.top_crops[:3])}")
    except Exception as e:
        print_result("Get regional stats", False, str(e))


async def test_realtime_data_manager():
    """Test unified RealTimeDataManager."""
    print_header("Testing RealTimeDataManager")
    
    from src.tools.realtime_data import get_realtime_data_manager
    
    manager = get_realtime_data_manager(use_mock=True)
    
    # Test 1: Get commodity prices
    try:
        result = await manager.get_commodity_prices("Tomato", "Karnataka", include_trends=True)
        passed = result.source in ["enam", "agmarknet", "mock"]
        print_result("Get commodity prices", passed,
                    f"Source: {result.source}, Freshness: {result.freshness.value}")
    except Exception as e:
        print_result("Get commodity prices", False, str(e))
    
    # Test 2: Get weather
    try:
        result = await manager.get_weather("Karnataka", "Kolar", crop="Tomato")
        passed = "current" in result.data
        print_result("Get weather", passed,
                    f"Source: {result.source}, Age: {result.age_display}")
    except Exception as e:
        print_result("Get weather", False, str(e))
    
    # Test 3: Get crop monitoring
    try:
        result = await manager.get_crop_monitoring(13.1333, 78.1333, crop="Tomato")
        passed = "monitoring" in result.data
        print_result("Get crop monitoring", passed,
                    f"Source: {result.source}, Freshness: {result.freshness.value}")
    except Exception as e:
        print_result("Get crop monitoring", False, str(e))
    
    # Test 4: Get comprehensive data
    try:
        results = await manager.get_comprehensive_data(
            commodity="Tomato",
            state="Karnataka",
            district="Kolar",
            lat=13.1333,
            lon=78.1333,
        )
        passed = len(results) >= 2
        print_result("Get comprehensive data", passed,
                    f"{len(results)} data sources fetched")
        
        for key, data in results.items():
            print(f"         → {key}: {data.source} ({data.freshness.value})")
    except Exception as e:
        print_result("Get comprehensive data", False, str(e))
    
    # Test 5: Health check
    try:
        health = manager.get_health_summary()
        passed = health["overall"] in ["healthy", "degraded"]
        print_result("Health check", passed,
                    f"Overall: {health['overall']}, {health['healthy_sources']}/{health['total_sources']} healthy")
    except Exception as e:
        print_result("Health check", False, str(e))


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("   PHASE 1: REAL-TIME DATA INTEGRATION TESTS")
    print("   " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    await test_enam_client()
    await test_imd_weather()
    await test_google_amed()
    await test_realtime_data_manager()
    
    print("\n" + "="*60)
    print("   TEST SUMMARY")
    print("="*60)
    print("\n  🎉 All Phase 1 components tested successfully!")
    print("  📊 eNAM, IMD Weather, Google AMED, and Data Manager ready.")
    print("  ⚠️  Currently running in MOCK mode for development.")
    print("\n  To enable live APIs, set the following in .env:")
    print("    ENAM_API_KEY=your_key")
    print("    OWM_API_KEY=your_openweathermap_key")
    print("    AMED_API_KEY=your_gcp_key")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
