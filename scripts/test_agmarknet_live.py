import asyncio
import sys
import logging
from loguru import logger
import os

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scrapers.agmarknet.client import AgmarknetScraper

logger.remove()
logger.add(sys.stdout, level="DEBUG")

async def test_scraper():
    logger.info("Initializing AgmarknetScraper...")
    scraper = AgmarknetScraper()
    
    # Test parameters
    state = "Karnataka"
    commodity = "Potato"
    district = "Kolar"
    
    logger.info(f"Running live test for {commodity} in {state} (District: {district})...")
    
    result = await scraper.scrape(
        state=state, 
        commodity=commodity, 
        district=district
    )
    
    logger.info("Scrape completed.")
    logger.info(f"Success: {result.success}")
    logger.info(f"URL: {result.url}")
    logger.info(f"Error: {result.error}")
    
    if hasattr(result, 'data') and result.data:
        logger.info(f"Extracted {len(result.data)} records.")
        for i, rec in enumerate(result.data[:5]):
            logger.info(f"Record {i+1}: {rec}")
    else:
        logger.warning("No data was extracted or result has no data attribute.")

    await scraper.close()
    
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(test_scraper())
