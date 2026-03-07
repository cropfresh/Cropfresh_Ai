"""Test Weather API connectivity."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx

key = os.getenv("WEATHER_API_KEY", "")
if not key:
    print("❌ WEATHER_API_KEY not set in .env")
    sys.exit(1)

print(f"🔗 Testing OpenWeatherMap API...")
print(f"   Key: {key[:8]}...")

r = httpx.get(
    f"https://api.openweathermap.org/data/2.5/weather",
    params={"q": "Bengaluru,IN", "appid": key, "units": "metric"},
)

print(f"   Status: {r.status_code}")
if r.status_code == 200:
    d = r.json()
    temp = d.get("main", {}).get("temp")
    desc = d.get("weather", [{}])[0].get("description")
    print(f"✅ Weather API OK! Bengaluru: {temp}°C, {desc}")
else:
    print(f"❌ Weather API FAILED: {r.text}")
    sys.exit(1)
