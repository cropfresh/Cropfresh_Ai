"""
Price Prediction Analysis
=========================
Analytical logic for price prediction (features, trends, rules).
"""

from datetime import datetime

import numpy as np

from src.tools.agmarknet import AgmarknetPrice

from .constants import HOLD_3D_DELTA, SEASONAL_CALENDAR, SELL_NOW_DELTA


class AnalysisMixin:
    """Mixin for price prediction analysis methods."""

    def _extract_features(
        self,
        history: list[AgmarknetPrice],
        commodity: str,
    ) -> dict[str, float]:
        """Build feature dict from price history."""
        prices = [p.modal_price_per_kg for p in history]
        n = len(prices)

        avg_7d = float(np.mean(prices[-7:])) if n >= 7 else float(np.mean(prices))
        avg_14d = float(np.mean(prices[-14:])) if n >= 14 else avg_7d
        avg_30d = float(np.mean(prices[-30:])) if n >= 30 else avg_14d

        # Momentum = relative price change over window
        momentum_7d = (prices[-1] - prices[-7]) / prices[-7] if n >= 7 and prices[-7] > 0 else 0.0
        momentum_30d = (prices[-1] - prices[-30]) / prices[-30] if n >= 30 and prices[-30] > 0 else 0.0

        month = datetime.now().month
        seasonal_multiplier = self._get_seasonal_multiplier(commodity, month)

        return {
            "avg_7d": avg_7d,
            "avg_14d": avg_14d,
            "avg_30d": avg_30d,
            "momentum_7d": momentum_7d,
            "momentum_30d": momentum_30d,
            "seasonal_multiplier": seasonal_multiplier,
            "latest_price": prices[-1],
        }

    def _rule_based_predict(self, features: dict[str, float], days_ahead: int) -> float:
        """
        Weighted rule-based prediction formula:
          predicted = 7d_avg × (1 + momentum × days/7) × seasonal_multiplier
        """
        avg_7d = features["avg_7d"]
        momentum = features["momentum_7d"]
        seasonal_mult = features["seasonal_multiplier"]

        raw = avg_7d * (1 + momentum * (days_ahead / 7.0))
        predicted = raw * seasonal_mult
        return max(predicted, 0.5)

    def _analyze_trend(self, history: list[AgmarknetPrice]) -> tuple[str, float]:
        """
        Linear regression on the last 14 days of modal prices.
        Returns (direction, strength)
        """
        prices = [p.modal_price_per_kg for p in history[-14:]]
        if len(prices) < 3:
            return "stable", 0.0

        x = np.arange(len(prices), dtype=float)
        slope = float(np.polyfit(x, prices, 1)[0])
        avg = float(np.mean(prices))
        rel_slope = slope / avg if avg > 0 else 0.0

        if rel_slope > 0.02:
            return "rising", min(1.0, rel_slope * 10)
        if rel_slope < -0.02:
            return "falling", min(1.0, abs(rel_slope) * 10)
        return "stable", 0.0

    def _get_seasonal_factor(self, commodity: str, month: int) -> str:
        """Return human-readable seasonal label for a crop in a given month."""
        multiplier = self._get_seasonal_multiplier(commodity, month)
        if multiplier >= 1.15:
            return "off_season"
        if multiplier <= 0.85:
            return "peak_harvest"
        return "normal"

    def _get_seasonal_multiplier(self, commodity: str, month: int) -> float:
        """Return numeric seasonal multiplier (default 1.0)."""
        crop_calendar = SEASONAL_CALENDAR.get(commodity.lower(), {})
        return crop_calendar.get(month, 1.0)

    def _generate_recommendation(
        self,
        current: float,
        predicted: float,
        trend: str,
        seasonal: str,
    ) -> str:
        """Determine sell/hold advice from price delta and trend context."""
        if current <= 0:
            return "hold_7d"

        delta = (predicted - current) / current

        if delta >= SELL_NOW_DELTA:
            return "sell_now"
        if delta >= HOLD_3D_DELTA:
            return "hold_3d"
        if trend == "rising" and seasonal == "off_season":
            return "hold_7d"
        return "hold_30d"

    def _explain_factors(
        self,
        features: dict[str, float],
        trend: str,
        seasonal: str,
        commodity: str,
    ) -> list[str]:
        """Generate human-readable list of price-driving factors."""
        factors: list[str] = []

        momentum = features.get("momentum_7d", 0.0)
        if momentum > 0.05:
            factors.append(f"Strong upward momentum (+{momentum * 100:.1f}% over 7 days).")
        elif momentum < -0.05:
            factors.append(f"Downward price pressure ({momentum * 100:.1f}% over 7 days).")
        else:
            factors.append("Price stable over last 7 days.")

        if trend == "rising":
            factors.append("14-day trend is rising — buyers increasing demand.")
        elif trend == "falling":
            factors.append("14-day trend is falling — market oversupply likely.")

        if seasonal == "off_season":
            factors.append(f"{commodity.title()} is in off-season — supply constrained, prices elevated.")
        elif seasonal == "peak_harvest":
            factors.append(f"{commodity.title()} is at peak harvest — high supply depresses prices.")

        seasonal_mult = features.get("seasonal_multiplier", 1.0)
        if seasonal_mult > 1.1:
            factors.append(f"Seasonal multiplier {seasonal_mult:.2f}× above baseline.")
        elif seasonal_mult < 0.9:
            factors.append(f"Seasonal multiplier {seasonal_mult:.2f}× below baseline.")

        return factors[:4]
