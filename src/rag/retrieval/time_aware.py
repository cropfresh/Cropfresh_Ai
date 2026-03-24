"""
Time-Aware Retriever — Freshness-boosted ranking (ADR-010 Phase 4).

Adjusts retrieval scores based on document freshness and query intent:
- Market price queries: boost documents <= 24h, heavily penalize > 7 days
- Weather queries: boost documents <= 48h, penalize > 5 days
- Agronomic knowledge: no time penalty (evergreen content)
- Scheme info: mild penalty for > 90 days old

Integrates with existing hybrid_search.py RRF fusion scores.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel


class FreshnessCategory(str, Enum):
    """Time sensitivity categories for queries."""

    MARKET = "market"        # Prices, demand — very time-sensitive
    WEATHER = "weather"      # Forecasts — moderately time-sensitive
    SCHEME = "scheme"        # Government schemes — somewhat evolving
    EVERGREEN = "evergreen"  # Agronomy, techniques — timeless


class TimeAwareResult(BaseModel):
    """A retrieval result with freshness-adjusted score."""

    doc_id: str = ""
    original_score: float = 0.0
    freshness_score: float = 1.0
    adjusted_score: float = 0.0
    age_hours: float = 0.0
    category: FreshnessCategory = FreshnessCategory.EVERGREEN


class TimeAwareRetriever:
    """Adjusts retrieval scores based on document age and query intent."""

    # ? Category-specific max age (in hours) and decay rates
    FRESHNESS_CONFIG: dict[FreshnessCategory, dict[str, float]] = {
        FreshnessCategory.MARKET: {
            "max_age_hours": 24.0,
            "decay_rate": 0.15,      # Aggressive decay for market data
            "floor": 0.2,            # Minimum freshness score
        },
        FreshnessCategory.WEATHER: {
            "max_age_hours": 48.0,
            "decay_rate": 0.10,
            "floor": 0.3,
        },
        FreshnessCategory.SCHEME: {
            "max_age_hours": 2160.0,  # 90 days
            "decay_rate": 0.005,
            "floor": 0.5,
        },
        FreshnessCategory.EVERGREEN: {
            "max_age_hours": 87600.0,  # 10 years
            "decay_rate": 0.0,
            "floor": 1.0,             # No penalty
        },
    }

    # ? Keywords for category classification
    MARKET_KEYWORDS = {
        "price", "rate", "cost", "mandi", "enam", "market",
        "sell", "buy", "auction", "quintal", "beele",  # Kannada
    }
    WEATHER_KEYWORDS = {
        "weather", "rain", "forecast", "temperature", "monsoon",
        "rainfall", "climate", "wind", "humidity", "havamana",
    }
    SCHEME_KEYWORDS = {
        "scheme", "subsidy", "government", "pm-kisan", "yojana",
        "benefit", "loan", "credit", "insurance",
    }

    def classify_query(self, query: str) -> FreshnessCategory:
        """Classify query into a freshness category."""
        lower = query.lower()

        if any(kw in lower for kw in self.MARKET_KEYWORDS):
            return FreshnessCategory.MARKET
        if any(kw in lower for kw in self.WEATHER_KEYWORDS):
            return FreshnessCategory.WEATHER
        if any(kw in lower for kw in self.SCHEME_KEYWORDS):
            return FreshnessCategory.SCHEME

        return FreshnessCategory.EVERGREEN

    def adjust_scores(
        self,
        documents: list[Any],
        query: str,
        current_time: float | None = None,
    ) -> list[TimeAwareResult]:
        """Adjust document scores based on freshness.

        Args:
            documents: Retrieved documents with .score and .metadata.
            query: Original query text.
            current_time: Current timestamp (for testing).

        Returns:
            Sorted list of TimeAwareResult with adjusted scores.
        """
        category = self.classify_query(query)
        config = self.FRESHNESS_CONFIG[category]
        now = current_time or time.time()
        results: list[TimeAwareResult] = []

        for doc in documents:
            original_score = getattr(doc, "score", 0.5)
            doc_id = getattr(doc, "id", "unknown")
            meta = getattr(doc, "metadata", {}) or {}

            # Extract document timestamp
            doc_time = meta.get("timestamp", meta.get("indexed_at", 0))
            if isinstance(doc_time, str):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(doc_time)
                    doc_time = dt.timestamp()
                except (ValueError, TypeError):
                    doc_time = 0

            age_hours = (now - float(doc_time)) / 3600 if doc_time else 0

            # Calculate freshness score with exponential decay
            if age_hours <= 0 or category == FreshnessCategory.EVERGREEN:
                freshness = 1.0
            else:
                decay = config["decay_rate"]
                freshness = max(
                    config["floor"],
                    1.0 / (1.0 + decay * age_hours),
                )

            adjusted = original_score * freshness

            results.append(TimeAwareResult(
                doc_id=doc_id,
                original_score=original_score,
                freshness_score=round(freshness, 4),
                adjusted_score=round(adjusted, 4),
                age_hours=round(age_hours, 1),
                category=category,
            ))

        # Sort by adjusted score descending
        results.sort(key=lambda r: r.adjusted_score, reverse=True)

        if results:
            logger.info(
                f"TimeAwareRetriever: category={category.value} | "
                f"docs={len(results)} | "
                f"top_adjusted={results[0].adjusted_score:.3f}"
            )

        return results
