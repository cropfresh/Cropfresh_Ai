"""
Price Prediction Constants
==========================
Seasonal multipliers and recommendation thresholds.
"""

# Karnataka seasonal price multipliers by crop and month (1=Jan … 12=Dec)
SEASONAL_CALENDAR: dict[str, dict[int, float]] = {
    "tomato": {
        1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 1.3, 6: 1.4,
        7: 1.2, 8: 1.0, 9: 0.9, 10: 0.85, 11: 0.8, 12: 0.9,
    },
    "onion": {
        1: 1.0, 2: 1.1, 3: 1.2, 4: 1.3, 5: 1.1, 6: 0.9,
        7: 0.8, 8: 0.7, 9: 0.8, 10: 1.0, 11: 0.7, 12: 0.8,
    },
    "potato": {
        1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.9, 6: 1.0,
        7: 1.1, 8: 1.2, 9: 1.1, 10: 1.0, 11: 0.95, 12: 1.0,
    },
    "cauliflower": {
        1: 0.9, 2: 0.8, 3: 0.7, 4: 0.8, 5: 1.1, 6: 1.2,
        7: 1.1, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.85, 12: 0.9,
    },
    "carrot": {
        1: 1.0, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.9, 6: 1.0,
        7: 1.0, 8: 1.1, 9: 1.2, 10: 1.1, 11: 1.0, 12: 1.0,
    },
    "okra": {
        1: 1.1, 2: 1.0, 3: 0.9, 4: 0.85, 5: 1.0, 6: 1.1,
        7: 1.2, 8: 1.1, 9: 1.0, 10: 0.95, 11: 0.9, 12: 1.0,
    },
}

# Labels describing the seasonal position of prices
SEASONAL_LABEL_MAP = {
    "peak_harvest": "Peak harvest season — high supply depresses prices.",
    "off_season": "Off-season — reduced supply supports higher prices.",
    "normal": "Normal seasonal conditions.",
}

# Recommendation thresholds
SELL_NOW_DELTA = 0.05    # predicted ≥ 5% above current → sell now
HOLD_3D_DELTA  = -0.03   # predicted 0–5% above current → hold 3 days
HOLD_7D_DELTA  = -0.08   # predicted < −3% → hold 7 days
