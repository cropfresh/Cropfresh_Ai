"""
Monitoring tools for scrapers.
Tracks health, anomalies, and logs missing properties.
"""
from typing import Any
from loguru import logger
from datetime import datetime
import smtplib # Placeholder for alert action

class ScraperMonitor:
    """Logs anomalies and raises alerts based on scraping stats."""

    def __init__(self, alert_threshold: int = 3):
        self.consecutive_failures = 0
        self.alert_threshold = alert_threshold

    def log_success(self, source_name: str, records_count: int):
        """Called when a scrape finishes successfully."""
        self.consecutive_failures = 0
        if records_count == 0:
            logger.warning(f"[{source_name}] Scrape succeeded but returned 0 records. Check site layout.")
        else:
            logger.info(f"[{source_name}] Successfully extracted {records_count} records.")

    def log_failure(self, source_name: str, error: Exception):
        """Called when a scraper fails completely."""
        self.consecutive_failures += 1
        logger.error(f"[{source_name}] Scrape failed: {error}")
        
        if self.consecutive_failures >= self.alert_threshold:
            self._trigger_alert(f"[{source_name}] Failed {self.consecutive_failures} times consecutively!")

    def log_anomaly(self, source_name: str, message: str):
        """Called for partial failures or unexpected data shapes."""
        logger.warning(f"[{source_name}] Anomaly detected: {message}")

    def _trigger_alert(self, message: str):
        """Internal method to fire off PagerDuty/Email/Slack alerts."""
        logger.critical(f"ALERT: {message}")
        # Send Email/Slack...
