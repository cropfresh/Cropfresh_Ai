import asyncio
import httpx
import re

async def find_endpoints():
    url = "https://agmarknet.gov.in/static/js/main.a1370aa2.js"
    
    async with httpx.AsyncClient(verify=False) as client:
        r = await client.get(url)
        content = r.text
        
        # Find endpoints starting with /v1/
        endpoints = re.findall(r'[\'"](/v1/[a-zA-Z0-9_-]+)[\'"]', content)
        
        print("Found endpoints:")
        for ep in sorted(list(set(endpoints))):
            print(ep)

asyncio.run(find_endpoints())
