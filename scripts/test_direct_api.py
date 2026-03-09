import requests
import json
import httpx
import asyncio

async def test_api():
    url = "https://api.agmarknet.gov.in/v1/daily-price-arrival/report"
    
    params = {
        "from_date": "2026-03-09",
        "to_date": "2026-03-09",
        "data_type": "100004", # Price (Arrival is 100005, Both is 100006)
        "group": "6",          # Vegetables
        "commodity": "23",     # Onion
        "state": "[20]",       # Maharashtra
        "district": "[100001]",# All districts
        "market": "[100002]",  # All markets
        "grade": "[100003]",   # All grades
        "variety": "[100007]", # All varieties
        "page": "1",
        "limit": "10"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://agmarknet.gov.in/"
    }
    
    print("Fetching API directly...")
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        with open("direct_api_response.json", "w", encoding="utf-8") as f:
            json.dump(response.json(), f, indent=2)
            
        print("Response saved to direct_api_response.json")

if __name__ == "__main__":
    asyncio.run(test_api())
