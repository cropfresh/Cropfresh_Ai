# F004: Quality Grading

## Overview
Computer vision-based crop quality assessment with HITL fallback and Digital Twin linkage for dispute resolution.

## Acceptance Criteria
- [x] Grade assignment (A+/A/B/C) from photo or description (Task 3)
- [x] Defect detection (bruise, worm_hole, colour_off, rot_spot, etc.) — via vision or rule-based (Task 3)
- [x] Confidence score for each grade (Task 3)
- [x] HITL trigger when confidence < 0.7 or grade A+ (Task 3)
- [x] Digital Twin departure snapshot creation (Task 10)
- [x] Arrival vs departure diff with liability recommendation (Task 10)

## Priority: P1 | Status: ✅ Implemented (Tasks 3 + 10)

## Related
- `src/agents/quality_assessment/` — CV-QG agent
- `src/agents/digital_twin/` — Digital Twin Engine (dispute diff)
- `src/api/services/order_service.py` — raise_dispute trigger
