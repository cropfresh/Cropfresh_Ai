"""
ADCL Agent — Seasonal Calendar
================================
Maps crops to planting/harvest seasons for major Indian agricultural
zones (Karnataka / South India baseline).

Harvest-fit labels (get_fit):
  'in_season'  — crop is actively harvested / market-ready this month
  'off_season' — not available / very low quality this month
  'year_round' — available throughout the year (or unknown crop)

Sow-fit labels (get_sow_fit):
  'ideal_sow'      — best month to sow/plant this crop
  'possible_sow'   — can be sown but not ideal
  'not_sow_season' — definitely not the time to plant

Data source: Karnataka Horticulture Department season calendars.
"""

# * ADCL SEASONAL MODULE
# NOTE: Static data; extend _CROP_CALENDAR as needed per district.

from __future__ import annotations


# * ═══════════════════════════════════════════════════════════════
# * Crop calendar — both sow and harvest months per crop
# * ═══════════════════════════════════════════════════════════════

#! Upgraded from flat _SEASON_MAP to structured dict with sow + harvest months.
# "sow" = best months to plant, "harvest" = months crop is market-ready.
_CROP_CALENDAR: dict[str, dict[str, set[int]]] = {
    # Vegetables (Kharif/Rabi)
    "tomato":       {"sow": {6, 7, 8, 9},        "harvest": {10, 11, 12, 1, 2, 3}},
    "onion":        {"sow": {6, 7, 8, 9, 10},    "harvest": {11, 12, 1, 2, 3, 4}},
    "potato":       {"sow": {8, 9, 10},           "harvest": {11, 12, 1, 2, 3}},
    "brinjal":      {"sow": {6, 7, 8, 9, 10},    "harvest": {1, 2, 3, 4, 5, 9, 10, 11, 12}},
    "capsicum":     {"sow": {7, 8, 9},            "harvest": {10, 11, 12, 1, 2, 3}},
    "cabbage":      {"sow": {7, 8, 9},            "harvest": {10, 11, 12, 1, 2, 3}},
    "cauliflower":  {"sow": {7, 8, 9},            "harvest": {10, 11, 12, 1, 2, 3}},
    "okra":         {"sow": {2, 3, 6, 7},         "harvest": {4, 5, 6, 7, 8, 9}},
    "beans":        {"sow": {7, 8, 9},            "harvest": {10, 11, 12, 1, 2}},
    "carrot":       {"sow": {8, 9, 10},           "harvest": {10, 11, 12, 1, 2}},
    "spinach":      {"sow": {8, 9, 10},           "harvest": {10, 11, 12, 1, 2, 3}},
    "radish":       {"sow": {8, 9, 10},           "harvest": {10, 11, 12, 1, 2}},
    "cucumber":     {"sow": {1, 2, 5, 6},         "harvest": {3, 4, 5, 6, 7, 8, 9}},
    "bitter gourd": {"sow": {2, 3, 6},            "harvest": {4, 5, 6, 7, 8, 9}},
    "ridge gourd":  {"sow": {2, 3, 6},            "harvest": {4, 5, 6, 7, 8}},
    "snake gourd":  {"sow": {2, 3, 6},            "harvest": {4, 5, 6, 7, 8}},
    # Fruits
    "mango":        {"sow": {7, 8},               "harvest": {3, 4, 5, 6}},
    "banana":       {"sow": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                     "harvest": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
    "papaya":       {"sow": {6, 7, 8, 9, 10},
                     "harvest": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
    "grapes":       {"sow": {6, 7, 8},            "harvest": {2, 3, 4, 5}},
    "watermelon":   {"sow": {12, 1, 2},           "harvest": {3, 4, 5, 6}},
    # Grains / Pulses / Spices
    "rice":         {"sow": {5, 6, 7},            "harvest": {9, 10, 11}},
    "maize":        {"sow": {5, 6, 7},            "harvest": {8, 9, 10}},
    "groundnut":    {"sow": {6, 7},               "harvest": {10, 11, 12, 1}},
    "sunflower":    {"sow": {11, 12, 1},          "harvest": {2, 3, 4, 5}},
    "turmeric":     {"sow": {5, 6, 7},            "harvest": {1, 2, 3}},
    "ginger":       {"sow": {4, 5, 6},            "harvest": {12, 1, 2, 3}},
}

# Crops that are available year-round (no seasonality penalty)
_YEAR_ROUND: set[str] = {"banana", "papaya", "coconut", "drumstick"}


class SeasonalCalendar:
    """
    Determines seasonal fit for a crop given the current month.

    Two methods:
      get_fit()     — harvest-season check (backward-compatible)
      get_sow_fit() — planting-season check (new in task35)

    Usage:
        cal = SeasonalCalendar()
        fit = cal.get_fit("tomato", month=12)          # 'in_season'
        sow = cal.get_sow_fit("tomato", month=8)       # 'ideal_sow'
        sow = cal.get_sow_fit("tomato", month=12)      # 'not_sow_season'
    """

    def get_fit(self, commodity: str, month: int) -> str:
        """
        Return harvest-season fit for commodity in given month.

        Args:
            commodity : Crop name (case-insensitive).
            month     : Month number 1–12.

        Returns:
            'in_season' | 'off_season' | 'year_round'
        """
        key = commodity.strip().lower()

        if key in _YEAR_ROUND:
            return "year_round"

        entry = _CROP_CALENDAR.get(key)
        if entry is None:
            # Unknown crop — treat as year_round to avoid penalising
            return "year_round"

        harvest_months = entry.get("harvest", set())
        if not harvest_months:
            return "year_round"

        return "in_season" if month in harvest_months else "off_season"

    def get_sow_fit(self, commodity: str, month: int) -> str:
        """
        Return sowing-season fit for commodity in given month.

        Args:
            commodity : Crop name (case-insensitive).
            month     : Month number 1–12.

        Returns:
            'ideal_sow' | 'possible_sow' | 'not_sow_season'
        """
        key = commodity.strip().lower()

        if key in _YEAR_ROUND:
            return "ideal_sow"

        entry = _CROP_CALENDAR.get(key)
        if entry is None:
            # ? Unknown crop — conservative: say it's fine to plant
            return "ideal_sow"

        sow_months = entry.get("sow", set())
        if not sow_months:
            return "ideal_sow"

        if month in sow_months:
            return "ideal_sow"

        # Check adjacent months (±1) for "possible" window
        adjacent = {(month % 12) + 1, ((month - 2) % 12) + 1}
        if adjacent & sow_months:
            return "possible_sow"

        return "not_sow_season"
