# Task 13: Implement APMC Live Mandi Scraper

> **Priority:** 🟠 P1 | **Phase:** 3 | **Effort:** 3–4 days  
> **Files:** `src/scrapers/agmarknet.py` (enhance), `src/scrapers/scraper_scheduler.py` [NEW]  
> **Score Target:** 9/10 — Daily automated mandi price capture for Karnataka

---

## 📌 Problem Statement

Real mandi prices are critical for AISP calculation, price prediction, and farmer advisory. eNAM registration is pending. Alternative: scrape Agmarknet.gov.in daily.

---

## 🔬 Research Findings

### Legal Compliance
- Agmarknet is a **government portal** — public data meant for farmer welfare
- Must respect `robots.txt` and rate-limit requests (max 1 req/sec)
- Use reasonable User-Agent, no aggressive scraping
- Target only commodity price bulletins (public data)

### Data Sources
| Source | URL | Data Available |
|--------|-----|----------------|
| Agmarknet | agmarknet.gov.in | Daily mandi prices, arrivals |
| eNAM | enam.gov.in | Real-time online trading prices |
| Data.gov.in | data.gov.in | Historical datasets (CSV) |

### Scraping Stack
- **Scrapling** (already in project) — stealth scraping with Camoufox
- **Selenium** fallback — for dynamic dropdowns
- **APScheduler** — daily job scheduling

---

## 🏗️ Implementation Spec

### Enhanced Agmarknet Scraper
```python
class AgmarknetScraper:
    """
    Scrapes daily commodity prices from Agmarknet.gov.in.
    
    Features:
    - Handles dynamic dropdowns (state → district → commodity)
    - Rate-limited (1 req/sec)
    - Retries with exponential backoff
    - Stores in price_history table
    - Graceful fallback to cached data
    """
    
    TARGET_URL = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    
    KARNATAKA_MANDIS = [
        "Bangalore", "Hubli", "Mysore", "Belgaum", 
        "Gulbarga", "Shimoga", "Mangalore", "Davangere",
    ]
    
    TARGET_COMMODITIES = [
        "Tomato", "Onion", "Potato", "Green Chilli",
        "Beans (French)", "Carrot", "Brinjal", "Cabbage",
        "Cauliflower", "Capsicum", "Ladies Finger",
    ]
    
    async def scrape_daily_prices(
        self,
        state: str = "Karnataka",
        date: Optional[date] = None,
    ) -> list[MandiPrice]:
        """
        Scrape today's prices for all target mandis × commodities.
        
        Returns list of MandiPrice records for insertion into price_history.
        """
    
    async def scrape_and_store(self) -> int:
        """Scrape + insert into DB. Returns count of records inserted."""
        prices = await self.scrape_daily_prices()
        count = await self.db.insert_price_history(prices)
        logger.info(f"Scraped {count} mandi prices for {datetime.now().date()}")
        return count
```

### Scheduler (`scraper_scheduler.py`)
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class ScraperScheduler:
    """
    Schedules daily scraping jobs.
    
    Schedule:
    - 10:00 AM IST: Agmarknet daily prices
    - 10:30 AM IST: eNAM prices (when registered)
    - 11:00 AM IST: IMD weather data
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    
    def start(self):
        self.scheduler.add_job(
            self.agmarknet_scraper.scrape_and_store,
            'cron', hour=10, minute=0,
            id='agmarknet_daily',
            misfire_grace_time=3600,
        )
        self.scheduler.start()
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Scrapes prices for 8+ Karnataka mandis | 25% |
| 2 | Handles 10+ commodities | 15% |
| 3 | Rate-limited (≤1 req/sec) with retries | 15% |
| 4 | Stores in price_history table | 20% |
| 5 | Daily scheduler runs at 10 AM IST | 15% |
| 6 | Graceful fallback when site is down | 10% |
