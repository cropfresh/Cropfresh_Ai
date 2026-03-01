# Quality Assessment Agent Specification

## Purpose

AI-powered produce grading (CV-QG) with HITL fallback and Digital Twin linkage for dispute-proof quality verification. Grades produce as A+/A/B/C, detects defects, estimates shelf life, and creates departure snapshots for dispute resolution.

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
- Vision pipeline requires ONNX models (yolov8n_agri_defects, dinov2_grade_classifier); falls back to rule-based when absent

## Dependencies

- `CropVisionPipeline` — vision_models.py (YOLOv8 defect detection + grade classifier)
- `DigitalTwinEngine` — `create_departure_twin()`, `compare_twin()` for dispute linkage

## Digital Twin Integration (Task 10)

- `create_departure_twin(listing_id, farmer_photos, agent_photos, quality_result, gps)` — Creates immutable snapshot at farm gate
- `compare_twin(twin_id, arrival_photos, arrival_gps)` — Compares departure vs arrival; returns DiffReport with liability recommendation
- Called by `OrderService._trigger_twin_diff()` when dispute is raised with arrival photos

## API

- `assess()` — Main grading entrypoint; returns QualityReport
- `execute()` — Supervisor contract; returns dict with grade, confidence, defects, hitl_required, digital_twin_linked
