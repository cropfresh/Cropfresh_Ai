# Task 35: Upgrade ADCL Agent — Fix Bugs & Add Sow-Season Logic

> **Priority:** 🟡 P2 | **Phase:** 4 (Upgrade) | **Effort:** 1 day  
> **Status:** ✅ **Completed — 2026-03-04**  
> **Files:** `src/agents/adcl/` (6 modules modified)  
> **Parent Task:** Task 12 (ADCL Agent — original implementation)

---

## 📌 Problem Statement

The ADCL Agent (Task 12) had three critical issues identified in research:

1. **Price trend misnomer**: `demand.py` calculated a volume-based trend but named it `price_trend`, misleading scoring and farmers.
2. **Normalization squashing**: Simple max-normalisation meant a single dominant crop could suppress all other scores below the 0.6 green-label threshold.
3. **Planting vs. harvesting seasonality**: The agent recommended planting crops that were currently being harvested, instead of crops whose sowing season is now.

---

## 🏗️ Changes Made

| Module        | Change                                                          |
| ------------- | --------------------------------------------------------------- |
| `models.py`   | Added `demand_trend` and `sow_season_fit` fields to `ADCLCrop`  |
| `demand.py`   | Renamed output to `demand_trend`; percentile-rank normalisation |
| `seasonal.py` | Added `get_sow_fit()` with sow/harvest calendar per crop        |
| `scoring.py`  | Green-label uses `demand_trend` + `sow_season_fit`              |
| `summary.py`  | Templates reference sowing context                              |
| `engine.py`   | Docstring updates for sow-season logic                          |

---

## ✅ Acceptance Criteria

| #   | Criterion                                                | Weight |
| --- | -------------------------------------------------------- | ------ |
| 1   | `demand_trend` and `price_trend` are separate fields     | 25%    |
| 2   | Percentile normalisation: multiple crops can score > 0.6 | 25%    |
| 3   | `sow_season_fit` correctly identifies planting windows   | 25%    |
| 4   | Green-label rule uses `demand_trend` + `sow_season_fit`  | 15%    |
| 5   | All existing + new tests pass                            | 10%    |
