# ADCL Agent — Specification

> **Package:** `src/agents/adcl/`  
> **Owner:** CropFresh AI  
> **Status:** Active (upgraded in Task 35)

---

## Overview

The **Adaptive Demand Crop List (ADCL)** agent generates weekly market intelligence for farmers, recommending which crops to **sow now** based on anticipated demand at harvest time.

## Architecture

```
┌──────────────┐     ┌────────────┐     ┌──────────────┐
│  engine.py   │────▶│ demand.py  │     │ seasonal.py  │
│ (orchestrator│     │ (aggregate │     │ (sow+harvest │
│  + DB I/O)   │     │  + trend)  │     │  calendar)   │
└──────┬───────┘     └────────────┘     └──────┬───────┘
       │                                       │
       ▼                                       ▼
┌──────────────┐     ┌────────────┐     ┌──────────────┐
│  scoring.py  │◀────│ models.py  │     │  summary.py  │
│ (green-label │     │ (ADCLCrop, │     │ (en/hi/kn    │
│  + recommend)│     │  Report)   │     │  templates)  │
└──────────────┘     └────────────┘     └──────────────┘
```

## Key Concepts

### Green Label Rule

A crop receives the **green label** (✅ recommended to grow) when ALL conditions are met:

- `demand_score > 0.6` (percentile-rank normalised)
- `demand_trend` is `'rising'` or `'stable'`
- `sow_season_fit` is NOT `'not_sow_season'`

### Demand Score (Percentile-Rank)

Each crop's score = fraction of other crops it beats or ties on `total_demand_kg`. This prevents a single dominant crop from suppressing all others.

### Seasonal Fit

Two dimensions:

- **Harvest fit** (`get_fit()`): Is the crop currently being harvested?
- **Sow fit** (`get_sow_fit()`): Is NOW a good time to plant this crop?

Green-label uses **sow fit** — recommending crops for planting, not harvest.

### Demand Trend vs Price Trend

- `demand_trend`: Volume-based (30d vs 60d order comparison)
- `price_trend`: Price-based (from PricePredictionAgent forecast)

## Module Reference

| Module        | Lines | Purpose                                     |
| ------------- | ----- | ------------------------------------------- |
| `models.py`   | ~95   | `ADCLCrop` + `WeeklyReport` dataclasses     |
| `demand.py`   | ~130  | Percentile-rank demand aggregation          |
| `seasonal.py` | ~130  | Sow + harvest calendar (Karnataka baseline) |
| `scoring.py`  | ~135  | Green-label rule + recommendations          |
| `summary.py`  | ~150  | Multi-language template + LLM summaries     |
| `engine.py`   | ~245  | Orchestrator + mock data + DB persistence   |

## API

```python
from src.agents.adcl import ADCLAgent, get_adcl_agent

agent = get_adcl_agent(db=my_db, price_agent=my_pricer, llm=my_llm)
report = await agent.generate_weekly_report(district="Bangalore")

# report.crops → list[ADCLCrop]
# report.summary_en / summary_hi / summary_kn → str
```
