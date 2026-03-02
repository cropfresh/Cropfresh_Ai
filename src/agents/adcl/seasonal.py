"""
ADCL Agent — Seasonal Calendar
================================
Maps crops to planting/harvest seasons for major Indian agricultural
zones (Karnataka / South India baseline).

Seasonal fit labels:
  'in_season'  — crop is actively harvested / market-ready this month
  'off_season' — not available / very low quality this month
  'year_round' — available throughout the year (or unknown crop)

Data source: Karnataka Horticulture Department season calendars.
"""

# * ADCL SEASONAL MODULE
# NOTE: Static data; extend _SEASON_MAP as needed per district.

from __future__ import annotations


# Month sets where the crop is IN season (Karnataka)
# Format: {crop_lowercase: set_of_month_numbers}
_SEASON_MAP: dict[str, set[int]] = {
    # Vegetables (Kharif/Rabi)
    "tomato":       {10, 11, 12, 1, 2, 3},
    "onion":        {11, 12, 1, 2, 3, 4},
    "potato":       {11, 12, 1, 2, 3},
    "brinjal":      {1, 2, 3, 4, 5, 9, 10, 11, 12},
    "capsicum":     {10, 11, 12, 1, 2, 3},
    "cabbage":      {10, 11, 12, 1, 2, 3},
    "cauliflower":  {10, 11, 12, 1, 2, 3},
    "okra":         {4, 5, 6, 7, 8, 9},
    "beans":        {10, 11, 12, 1, 2},
    "carrot":       {10, 11, 12, 1, 2},
    "spinach":      {10, 11, 12, 1, 2, 3},
    "radish":       {10, 11, 12, 1, 2},
    "cucumber":     {3, 4, 5, 6, 7, 8, 9},
    "bitter gourd": {4, 5, 6, 7, 8, 9},
    "ridge gourd":  {4, 5, 6, 7, 8},
    "snake gourd":  {4, 5, 6, 7, 8},
    # Fruits
    "mango":        {3, 4, 5, 6},
    "banana":       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  # year-round but peaks
    "papaya":       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    "grapes":       {2, 3, 4, 5},
    "watermelon":   {3, 4, 5, 6},
    # Grains / Pulses
    "rice":         {9, 10, 11},
    "maize":        {8, 9, 10},
    "groundnut":    {10, 11, 12, 1},
    "sunflower":    {2, 3, 4, 5},
    "turmeric":     {1, 2, 3},
    "ginger":       {12, 1, 2, 3},
}

# Crops that are available year-round (no seasonality penalty)
_YEAR_ROUND: set[str] = {"banana", "papaya", "coconut", "drumstick"}


class SeasonalCalendar:
    """
    Determines seasonal fit for a crop given the current month.

    Usage:
        cal = SeasonalCalendar()
        fit = cal.get_fit("tomato", month=12)      # 'in_season'
        fit = cal.get_fit("mango", month=12)       # 'off_season'
        fit = cal.get_fit("unknown_crop", month=6) # 'year_round'
    """

    def get_fit(self, commodity: str, month: int) -> str:
        """
        Return seasonal fit for commodity in given month.

        Args:
            commodity : Crop name (case-insensitive).
            month     : Month number 1–12.

        Returns:
            'in_season' | 'off_season' | 'year_round'
        """
        key = commodity.strip().lower()

        if key in _YEAR_ROUND:
            return "year_round"

        in_season_months = _SEASON_MAP.get(key)
        if in_season_months is None:
            # Unknown crop — treat as year_round to avoid penalising
            return "year_round"

        return "in_season" if month in in_season_months else "off_season"
