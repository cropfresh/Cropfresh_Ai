"""
Real-Time Knowledge Injection
=============================
Module for ingesting streaming news, weather alerts, and market updates.

Features:
- Streaming news aggregation (Google News RSS / NewsAPI)
- Market alert generation
- Weather advisory processing
- Knowledge graph injection (converting stream 2 graph)

Author: CropFresh AI Team
Version: 1.0.0
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid

from loguru import logger
from pydantic import BaseModel, Field


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RealTimeUpdate(BaseModel):
    """Normalized real-time update structure."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str  # 'news', 'price_alert', 'weather_alert', 'scheme'
    title: str
    content: str
    source: str
    url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: AlertSeverity = AlertSeverity.INFO
    entities: List[str] = Field(default_factory=list)  # Extracted entities


class NewsStreamer:
    """
    Ingests news from configured sources (RSS, APIs) and normalizes them.
    """
    
    def __init__(self, sources: List[str] = None):
        self.sources = sources or ["https://news.google.com/rss/search?q=agriculture+india"]
        logger.info(f"NewsStreamer initialized with {len(self.sources)} sources")
    
    async def fetch_latest_news(self, limit: int = 5) -> List[RealTimeUpdate]:
        """Fetch news from configured sources."""
        # Mocking news fetch for now - in prod use feedparser or httpx
        logger.info("Fetching latest agricultural news...")
        
        # Simulated news items
        mock_news = [
            RealTimeUpdate(
                type="news",
                title="Government announces MSP hike for Kharif crops",
                content="The Cabinet Committee on Economic Affairs has approved an increase in MSP for all mandated Kharif crops.",
                source="AgriNews India",
                severity=AlertSeverity.INFO,
                entities=["MSP", "Kharif", "Government"]
            ),
            RealTimeUpdate(
                type="news",
                title="Heavy rains predicted in Karnataka coffee belt",
                content="IMD has issued a yellow alert for Kodagu and Chikmagalur districts.",
                source="Weather Daily",
                severity=AlertSeverity.WARNING,
                entities=["Karnataka", "Rain", "Coffee"]
            ),
            RealTimeUpdate(
                type="news",
                title="Tomato prices crash in Kolar market due to excess supply",
                content="Prices dropped to 5 Rs/kg as arrivals doubled compared to last week.",
                source="Market Watch",
                severity=AlertSeverity.CRITICAL,
                entities=["Tomato", "Kolar", "Price"]
            )
        ]
        
        return mock_news[:limit]


class MarketAlertSystem:
    """
    Generates alerts based on price thresholds and market anomalies.
    """
    
    def __init__(self, threshold_pct: float = 10.0):
        self.threshold_pct = threshold_pct
    
    async def check_price_alerts(self, commodity: str, current_price: float, avg_price: float) -> Optional[RealTimeUpdate]:
        """Check if price deviation warrants an alert."""
        
        if current_price == 0 or avg_price == 0:
            return None
            
        diff_pct = ((current_price - avg_price) / avg_price) * 100
        
        if abs(diff_pct) >= self.threshold_pct:
            direction = "surged" if diff_pct > 0 else "crashed"
            severity = AlertSeverity.CRITICAL if abs(diff_pct) > 20 else AlertSeverity.WARNING
            
            return RealTimeUpdate(
                type="price_alert",
                title=f"Price Alert: {commodity} {direction} by {abs(diff_pct):.1f}%",
                content=f"Current price: ₹{current_price}, Moving Average: ₹{avg_price}. This is a significant deviation.",
                source="CropFresh Market Monitor",
                severity=severity,
                entities=[commodity, "Price Alert"]
            )
        
        return None



class WeatherAdvisorySystem:
    """
    Generates automated crop advisories based on weather forecasts.
    """
    
    def __init__(self):
        # crop_specific_rules: {crop: {condition: advisory}}
        self.rules = {
            "Tomato": {
                "Rain > 20mm": "Heavy rain alert: Ensure proper drainage to prevent waterlogging. Delay spraying chemicals.",
                "Temp > 35C": "High temperature alert: Provide light irrigation to reduce heat stress.",
                "Humidity > 90%": "High humidity alert: Watch out for Early Blight symptoms."
            },
            "Potato": {
                "Temp < 10C": "Cold wave alert: Risk of frost damage. Irrigate fields in evening.",
                "Rain > 10mm": "Rain alert: Postpone harvesting to avoid tuber rotting."
            }
        }
        logger.info("WeatherAdvisorySystem initialized")
        
    async def generate_advisories(self, crop: str, weather_data: Dict[str, Any]) -> List[RealTimeUpdate]:
        """Generate advisories based on weather conditions."""
        advisories = []
        crop_rules = self.rules.get(crop, {})
        
        # Mock logic to check rules against weather_data
        # weather_data expected format: {"temp": 36, "rain": 5, "humidity": 80}
        
        temp = weather_data.get("temp", 25)
        rain = weather_data.get("rain", 0)
        humidity = weather_data.get("humidity", 60)
        
        if crop == "Tomato":
            if temp > 35:
                advisories.append(self._create_advisory(crop, "High Temp", self.rules["Tomato"]["Temp > 35C"], AlertSeverity.WARNING))
            if rain > 20:
                advisories.append(self._create_advisory(crop, "Heavy Rain", self.rules["Tomato"]["Rain > 20mm"], AlertSeverity.CRITICAL))
            if humidity > 90:
                advisories.append(self._create_advisory(crop, "High Humidity", self.rules["Tomato"]["Humidity > 90%"], AlertSeverity.INFO))
                
        elif crop == "Potato":
            if temp < 10:
                advisories.append(self._create_advisory(crop, "Cold Wave", self.rules["Potato"]["Temp < 10C"], AlertSeverity.WARNING))
                
        return advisories
    
    def _create_advisory(self, crop: str, condition: str, advice: str, severity: AlertSeverity) -> RealTimeUpdate:
        return RealTimeUpdate(
            type="weather_advisory",
            title=f"{crop} Advisory: {condition}",
            content=advice,
            source="CropFresh Agro-Meteorology Dept",
            severity=severity,
            entities=[crop, "Weather"]
        )


class SchemeCrawler:
    """
    Crawls government portals for new agricultural schemes and updates.
    """
    
    def __init__(self):
        self.sources = [
            "https://agricoop.nic.in",
            "https://pmkisan.gov.in"
        ]
        logger.info("SchemeCrawler initialized")
        
    async def check_for_updates(self) -> List[RealTimeUpdate]:
        """Check for new schemes or updates."""
        # Simulated crawler
        updates = [
            RealTimeUpdate(
                type="scheme",
                title="PM-Kisan 16th Installment Released",
                content="The 16th installment of PM-Kisan Samman Nidhi has been released. Check status online.",
                source="PM-Kisan Portal",
                url="https://pmkisan.gov.in",
                severity=AlertSeverity.INFO,
                entities=["PM-Kisan", "Subsidy"]
            ),
            RealTimeUpdate(
                type="scheme",
                title="New Drone Didi Scheme for Fertilizer Spraying",
                content="Subsidy up to 80% for women self-help groups to buy agricultural drones.",
                source="Ministry of Agriculture",
                severity=AlertSeverity.INFO,
                entities=["Drone", "Subsidy", "Fertilizer"]
            )
        ]
        return updates


class KnowledgeInjector:
    """
    Injects real-time updates into the RAG context or vector store.
    """
    
    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.fresh_updates: List[RealTimeUpdate] = []
        
        # Sub-components
        self.news_streamer = NewsStreamer()
        self.market_alerts = MarketAlertSystem()
        self.weather_advisory = WeatherAdvisorySystem()
        self.scheme_crawler = SchemeCrawler()
    
    async def fetch_all_updates(self) -> int:
        """Fetch updates from all sources and ingest."""
        updates = []
        
        # News
        news = await self.news_streamer.fetch_latest_news()
        updates.extend(news)
        
        # Schemes
        schemes = await self.scheme_crawler.check_for_updates()
        updates.extend(schemes)
        
        # Mock weather check for advisory generation
        advisories = await self.weather_advisory.generate_advisories("Tomato", {"temp": 36, "rain": 0, "humidity": 65})
        updates.extend(advisories)
        
        await self.ingest_updates(updates)
        return len(updates)
    
    async def ingest_updates(self, updates: List[RealTimeUpdate]):
        """Store updates in short-term memory."""
        self.fresh_updates.extend(updates)
        # Sort by freshness
        self.fresh_updates.sort(key=lambda x: x.timestamp, reverse=True)
        # Keep only last 50 updates
        self.fresh_updates = self.fresh_updates[:50]
        logger.info(f"Ingested {len(updates)} real-time updates")
        
    def get_context_injection(self, query: str) -> str:
        """
        Get relevant realtime context string to inject into LLM prompt.
        Simple keyword matching for now.
        """
        relevant = []
        query_terms = set(query.lower().split())
        
        for update in self.fresh_updates:
            # Check overlap with entities or title words
            update_terms = set(update.title.lower().split())
            update_entities = {e.lower() for e in update.entities}
            
            if query_terms.intersection(update_terms) or query_terms.intersection(update_entities):
                relevant.append(f"[{update.type.upper()}] {update.timestamp.strftime('%H:%M')}: {update.title} - {update.content}")
        
        if not relevant:
            return ""
            
        return "\n--- REAL-TIME UPDATES ---\n" + "\n".join(relevant) + "\n-------------------------\n"


# Factory
def create_knowledge_injector() -> KnowledgeInjector:
    return KnowledgeInjector()
