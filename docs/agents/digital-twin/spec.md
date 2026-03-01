# Digital Twin Engine Specification

## Purpose

Creates immutable departure snapshots of produce quality at the farm gate and compares them against buyer-submitted arrival photos to generate AI-powered diff reports for dispute resolution. Critical for the <2% dispute target.

## Module

`src/agents/digital_twin/`

## Key Components

| File | Purpose |
|------|---------|
| `models.py` | DigitalTwin, ArrivalData, DiffReport dataclasses |
| `diff_analysis.py` | SSIM → perceptual hash → rule-based similarity chain |
| `liability.py` | 6-rule liability matrix |
| `engine.py` | DigitalTwinEngine — create_departure_twin(), compare_arrival(), generate_diff_report() |

## API

- `create_departure_twin(listing_id, farmer_photos, agent_photos, quality_result, gps)` → DigitalTwin
- `compare_arrival(twin_id, arrival_photos, arrival_gps)` → DiffReport
- `generate_diff_report(departure_twin, arrival_data)` → DiffReport

## Liability Matrix

| Priority | Condition | Outcome |
|----------|-----------|---------|
| 1 | No arrival photos | Claim rejected (0%) |
| 2 | Quantity mismatch > 5% | Shared (farmer + hauler) |
| 3 | No quality degradation | No liability |
| 4 | Grade drop + transit > 6h | Hauler (cold-chain) |
| 5 | Grade drop + transit < 2h | Farmer (pre-existing) |
| 6 | Grade drop + transit 2–6h | Shared |

## Integration

- **QualityAssessmentAgent** — `compare_twin()`, `create_departure_twin()` delegate to engine
- **OrderService** — `_trigger_twin_diff()` calls engine or QA agent when dispute raised with arrival_photos + departure_twin_id
- **postgres_client** — `get_digital_twin()`, `update_dispute_diff_report()`

## Tests

`tests/unit/test_digital_twin.py` — 42 unit tests
