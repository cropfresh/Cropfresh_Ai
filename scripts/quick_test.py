"""Check states with data today."""
import os
import httpx
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('AGMARKNET_API_KEY', '')
url = 'https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070'

# Get any records
params = {
    'api-key': api_key, 
    'format': 'json', 
    'limit': 20,
}

resp = httpx.get(url, params=params, timeout=30)
data = resp.json()

print(f"Total records today: {data.get('total', 0)}")
print(f"\nSample prices from today:")
states = set()
for r in data.get('records', []):
    states.add(r['state'])
    print(f"  [{r['state']}] {r['commodity']} @ {r['market']}: Rs.{r['modal_price']}/q")

print(f"\nStates with data: {sorted(states)}")
