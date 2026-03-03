# F004: Quality Grading

## Overview

Computer vision-based crop quality assessment with HITL fallback and Digital Twin linkage for dispute resolution. Phase 1 implemented a rule-based scaffold; Phase 2 integrates real ONNX model inference (YOLOv8, DINOv2, ResNet50).

## Acceptance Criteria

### Phase 1 — Scaffold ✅ Complete (Tasks 3 + 10)

- [x] Grade assignment (A+/A/B/C) from photo or description (Task 3)
- [x] Defect detection (bruise, worm_hole, colour_off, rot_spot, etc.) — via vision or rule-based (Task 3)
- [x] Confidence score for each grade (Task 3)
- [x] HITL trigger when confidence < 0.7 or grade A+ (Task 3)
- [x] Digital Twin departure snapshot creation (Task 10)
- [x] Arrival vs departure diff with liability recommendation (Task 10)

### Phase 2 — Real CV Model Inference 🔴 Pending (Tasks 31–33)

- [ ] YOLOv26 ONNX real defect detection with bounding boxes + NMS (Task 31)
- [ ] DINOv2 ViT-S/14 ONNX grade classification with softmax confidence (Task 32)
- [ ] YOLO + DINOv2 ensemble override (critical defects downgrade premium grades) (Task 32)
- [ ] ResNet50 departure↔arrival cosine similarity for Digital Twin verification (Task 33)
- [ ] Substitution fraud flag when similarity < 0.50 (Task 33)
- [ ] Assessment mode returns `"vision"` when real models loaded (Tasks 31–32)
- [ ] Full pipeline CPU latency ≤ 500ms per image (Tasks 31–32)

## Priority: P0 | Status: 🟡 Phase 1 Complete — Phase 2 Real Models Pending

## Related

- `src/agents/quality_assessment/` — CV-QG agent + `vision_models.py`
- `src/agents/digital_twin/` — Digital Twin Engine + `similarity.py` (new, Task 33)
- `src/api/services/order_service.py` — raise_dispute trigger
- `models/vision/` — ONNX model artifacts (gitignored)
- `tracking/tasks/task31.md` — YOLO real inference spec
- `tracking/tasks/task32.md` — DINOv2 classifier spec
- `tracking/tasks/task33.md` — ResNet similarity engine spec
