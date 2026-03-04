"""
ML Price Forecaster
===================
Lightweight time-series price forecaster for agricultural commodities.

Uses a 3-step ensemble approach with stdlib + numpy only (no sklearn required
at inference time).  When scikit-learn is present a LinearRegression trend is
added as a fourth signal; otherwise it degrades gracefully to the other 3.

Strategy (MAP-REDUCE style):
  1. Exponential Weighted Moving Average (EWMA)   – quick recent signal
  2. Holt's double-exponential smoothing            – captures linear trend
  3. Seasonal decomposition residual adjustment    – monthly seasonality
  4. LinearRegression extrapolation (sklearn guard) – long-range trend anchor

All four forecasts are blended with configurable weights.

Author: CropFresh AI Team
Version: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Optional

from loguru import logger


# ── Domain model ────────────────────────────────────────────────────────────

@dataclass
class PriceSample:
    """One observed price with a timestamp."""
    date: datetime
    price_per_kg: float


@dataclass
class ForecastResult:
    """Output of the ML forecasting step."""
    commodity: str
    location: str
    current_avg_price: float
    forecasted_prices: list[float]       # day-by-day, horizon length
    horizon_days: int
    confidence: float                    # 0.0 – 1.0
    trend_direction: str                 # "rising" | "falling" | "stable"
    trend_pct_change: float              # expected % change over horizon
    volatility_index: float              # coefficient of variation, capped 0-1
    models_used: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _ewma(prices: list[float], alpha: float = 0.3) -> list[float]:
    """
    Exponential weighted moving average.
    alpha ∈ (0, 1]; higher = more weight on recent observations.
    """
    smoothed = [prices[0]]
    for p in prices[1:]:
        smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
    return smoothed


def _holt_double_exp(
    prices: list[float],
    alpha: float = 0.4,
    beta: float = 0.2,
    horizon: int = 7,
) -> list[float]:
    """
    Holt's double-exponential smoothing – captures level + linear trend.
    Returns `horizon` future values.
    """
    if len(prices) < 2:
        return [prices[-1]] * horizon

    level = prices[0]
    trend = prices[1] - prices[0]

    for p in prices[1:]:
        new_level = alpha * p + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level

    return [level + (i + 1) * trend for i in range(horizon)]


def _seasonal_factor(commodity: str, target_date: datetime) -> float:
    """
    Monthly seasonal multiplier sourced from Karnataka historical patterns.
    Returns 1.0 for commodities / months with no known seasonal bias.
    """
    #! These factors are Karnataka-specific; update if expanding to other states.
    table: dict[str, dict[int, float]] = {
        "tomato":      {5: 1.30, 6: 1.20, 7: 1.15, 11: 0.90, 12: 0.85},
        "onion":       {11: 0.80, 12: 0.82, 1: 0.88, 5: 1.10, 6: 1.15},
        "potato":      {2: 0.90, 3: 0.88, 9: 1.08, 10: 1.12},
        "cabbage":     {12: 0.85, 1: 0.86, 4: 1.10, 5: 1.12},
        "cauliflower": {12: 0.86, 1: 0.88, 4: 1.10, 5: 1.15},
    }
    factors = table.get(commodity.strip().lower(), {})
    return factors.get(target_date.month, 1.0)


def _linear_trend_forecast(prices: list[float], horizon: int) -> Optional[list[float]]:
    """
    Optional sklearn LinearRegression extrapolation.
    Returns None if sklearn is unavailable (graceful degradation).
    """
    try:
        from sklearn.linear_model import LinearRegression  # type: ignore
        import numpy as np  # type: ignore

        x = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices)
        model = LinearRegression().fit(x, y)
        future_x = np.arange(len(prices), len(prices) + horizon).reshape(-1, 1)
        return model.predict(future_x).tolist()
    except ImportError:
        return None


def _blend_forecasts(
    forecasts: dict[str, list[float]],
    weights: Optional[dict[str, float]] = None,
) -> list[float]:
    """
    Weighted blend of multiple per-day forecasts.
    Default weights balance stability vs. recency.
    """
    default_w: dict[str, float] = {
        "holt": 0.40,
        "ewma": 0.30,
        "linear": 0.20,
        "seasonal": 0.10,
    }
    w = {k: weights.get(k, default_w.get(k, 0.0)) for k in forecasts} if weights else default_w

    # Normalize weights to only the models that are present
    present = {k: w[k] for k in forecasts if k in w}
    total = sum(present.values()) or 1.0
    norm_w = {k: v / total for k, v in present.items()}

    horizon = max(len(v) for v in forecasts.values())
    blended: list[float] = []
    for i in range(horizon):
        day_val = sum(
            forecasts[k][i] * norm_w[k]
            for k in present
            if i < len(forecasts[k])
        )
        blended.append(day_val)
    return blended


# ── Main forecaster ──────────────────────────────────────────────────────────

class PriceForecaster:
    """
    Ensemble time-series forecaster for crop prices.

    Usage::

        forecaster = PriceForecaster()
        result = forecaster.forecast(samples, "tomato", "Kolar", horizon=7)
    """

    DEFAULT_MIN_SAMPLES = 5
    DEFAULT_HORIZON = 7
    EWMA_ALPHA = 0.3
    HOLT_ALPHA = 0.4
    HOLT_BETA  = 0.2

    def forecast(
        self,
        samples: list[PriceSample],
        commodity: str,
        location: str,
        horizon: int = DEFAULT_HORIZON,
    ) -> ForecastResult:
        """
        Forecast prices for `horizon` future days from historical `samples`.

        Args:
            samples:   Ordered list of historical price observations (oldest first).
            commodity: Crop name — used for seasonal adjustment.
            location:  Market location (metadata only at this stage).
            horizon:   Number of future days to forecast.

        Returns:
            ForecastResult with day-by-day forecasts and trend metadata.
        """
        if len(samples) < self.DEFAULT_MIN_SAMPLES:
            logger.warning(
                f"Fewer than {self.DEFAULT_MIN_SAMPLES} samples for {commodity} — "
                "forecast may be unreliable"
            )

        samples_sorted = sorted(samples, key=lambda s: s.date)
        prices = [s.price_per_kg for s in samples_sorted]

        current_avg = mean(prices[-7:]) if len(prices) >= 7 else mean(prices)

        # ── Step 1: EWMA (recent signal)
        ewma_smooth = _ewma(prices, alpha=self.EWMA_ALPHA)
        #  * Project forward by propagating last smoothed value as a flat line
        ewma_forecast = [ewma_smooth[-1]] * horizon

        # ── Step 2: Holt double-exp (trend signal)
        holt_forecast = _holt_double_exp(prices, self.HOLT_ALPHA, self.HOLT_BETA, horizon)

        # ── Step 3: Seasonal adjustment overlay
        base_date = (samples_sorted[-1].date if samples_sorted
                     else datetime.now())
        seasonal_multipliers = [
            _seasonal_factor(commodity, base_date + timedelta(days=i + 1))
            for i in range(horizon)
        ]
        # Apply seasonal multiplier to the Holt baseline
        seasonal_forecast = [holt_forecast[i] * seasonal_multipliers[i] for i in range(horizon)]

        # ── Step 4: Linear regression (optional sklearn)
        linear_forecast = _linear_trend_forecast(prices, horizon)
        models_used = ["ewma", "holt", "seasonal"]
        if linear_forecast:
            models_used.append("linear")

        # ── Blend
        forecast_dict: dict[str, list[float]] = {
            "ewma": ewma_forecast,
            "holt": holt_forecast,
            "seasonal": seasonal_forecast,
        }
        if linear_forecast:
            forecast_dict["linear"] = linear_forecast

        blended = _blend_forecasts(forecast_dict)
        # Guard against negative prices
        blended = [max(p, 0.0) for p in blended]

        # ── Metadata
        volatility = pstdev(prices) / mean(prices) if mean(prices) > 0 else 0.0
        volatility_index = min(max(volatility, 0.0), 1.0)

        forecasted_end = blended[-1] if blended else current_avg
        pct_change = ((forecasted_end - current_avg) / current_avg * 100) if current_avg > 0 else 0.0

        if pct_change > 3:
            trend = "rising"
        elif pct_change < -3:
            trend = "falling"
        else:
            trend = "stable"

        # Confidence reduces with high volatility and low sample count
        sample_penalty = max(0.0, 1.0 - (self.DEFAULT_MIN_SAMPLES - len(samples)) / 10)
        confidence = round((1.0 - volatility_index * 0.5) * sample_penalty, 2)
        confidence = min(max(confidence, 0.1), 0.95)

        return ForecastResult(
            commodity=commodity,
            location=location,
            current_avg_price=round(current_avg, 2),
            forecasted_prices=[round(p, 2) for p in blended],
            horizon_days=horizon,
            confidence=confidence,
            trend_direction=trend,
            trend_pct_change=round(pct_change, 2),
            volatility_index=round(volatility_index, 4),
            models_used=models_used,
        )

    def forecast_from_raw(
        self,
        prices_list: list[float],
        commodity: str,
        location: str,
        horizon: int = DEFAULT_HORIZON,
        start_date: Optional[datetime] = None,
    ) -> ForecastResult:
        """Convenience wrapper: build PriceSamples from a raw list of prices."""
        base = start_date or datetime.now()
        samples = [
            PriceSample(date=base - timedelta(days=len(prices_list) - i - 1), price_per_kg=p)
            for i, p in enumerate(prices_list)
        ]
        return self.forecast(samples, commodity, location, horizon)
