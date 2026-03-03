# Quality Assessment Agent Specification

## Purpose

AI-powered produce grading (CV-QG) with HITL fallback and Digital Twin linkage for dispute-proof quality verification. Grades produce as A+/A/B/C, detects defects, estimates shelf life, and creates departure snapshots for dispute resolution.

## Two-Phase Architecture

### Phase 1 (Current — Rules & Stubs)

- Rule-based text keyword matching for defect detection
- Heuristic grade assignment from defect count
- Fallback when ONNX model files absent

### Phase 2 (Tasks 31–33 — Real ONNX Inference)

- **Stage 1:** YOLOv26n ONNX — real defect detection with bounding boxes + NMS post-processing
- **Stage 2:** DINOv2 ViT-S/14 ONNX — softmax grade probabilities, ensemble override with YOLO
- **Stage 3:** ResNet50 contrastive — Digital Twin departure↔arrival cosine similarity, substitution fraud flag

## Inputs

- `listing_id` — UUID of the crop listing
- `commodity` — Crop name (e.g. Tomato, Onion)
- `description` — Text description of produce condition (optional)
- `image_b64` — Base64-encoded photo (optional; triggers vision pipeline when present)
- `require_upgrade_review` — Boolean; forces HITL when farmer requests grade upgrade

## Outputs

- `GradeAssessment` — grade, confidence, defects_detected, defect_count, shelf_life_days, hitl_required, reasoning, assessment_id
- `QualityReport` — assessment + image_count + method (vision | rule_based | manual) + digital_twin_linked

## Constraints

- HITL required when: confidence < 0.7, grade is A+, or farmer requests upgrade
- Grade must be one of: A+, A, B, C
- Phase 2: Vision pipeline requires ONNX models in `models/vision/` (yolov8n, dinov2, resnet50); falls back to rule-based when absent
- Phase 2: Full pipeline CPU latency target ≤ 500ms/image

## Dependencies

- `CropVisionPipeline` — `vision_models.py` (YOLOv8 defect detection + DINOv2 grade classifier)
- `ResNetSimilarityEngine` — `src/agents/digital_twin/similarity.py` [Task 33, NEW] (departure↔arrival verification)
- `DigitalTwinEngine` — `create_departure_twin()`, `compare_twin()` for dispute linkage

## Digital Twin Integration (Task 10 + Task 33)

- `create_departure_twin(listing_id, farmer_photos, agent_photos, quality_result, gps)` — Creates immutable snapshot at farm gate
- `compare_twin(twin_id, arrival_photos, arrival_gps)` — Compares departure vs arrival; returns DiffReport with liability recommendation
- **Task 33:** ResNet50 cosine similarity replaces heuristic hash-based `compute_similarity()`
- Called by `OrderService._trigger_twin_diff()` when dispute is raised with arrival photos

## API

- `assess()` — Main grading entrypoint; returns QualityReport
- `execute()` — Supervisor contract; returns dict with grade, confidence, defects, hitl_required, digital_twin_linked

## Related Tasks

- Task 3 — Initial scaffold (complete)
- Task 10 — Digital Twin Engine (complete)
- **Task 31** — YOLOv8 real defect detection (pending)
- **Task 32** — DINOv2 grade classifier (pending)
- **Task 33** — ResNet50 Digital Twin similarity (pending)
