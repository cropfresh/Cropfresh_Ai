# Sprint Retrospective — Vision Subsystem

## 🟢 What Went Well

- **Two-Stage Quality Pipeline**: Successfully implemented a robust architecture (`CropVisionPipeline`) separating defect localization (YOLOv26) and holistic grading (DINOv2) with well-defined ensemble rules.
- **Graceful Degradation**: Built-in safety nets allow the system to gracefully fall back to keyword-based or heuristic grading (`detect_from_description`, `_grade_from_defect_count`) when ONNX models are unavailable, preventing pipeline crashes.
- **Digital Twin ResNet Similarity**: The `ResNetSimilarityEngine` handles visual verification robustly with fallback options (pHash) and handles cross-modal URL fetching efficiently.
- **Ensemble Override Logic**: The YOLO ensemble override smartly prevents DINOv2 hallucinations by enforcing strict downgrade caps (`_CRITICAL_DOWNGRADE_CAP`) when critical physical defects are identified.

## 🟡 What Could Improve

- **Dependency on `cv2.dnn.NMSBoxes`**: The YOLO detector relies on OpenCV's DNN module for Non-Maximum Suppression, which can be computationally heavy or cause install issues in headless/serverless environments.
- **Hardcoded Constants**: Several grading thresholds (`_CONF_THRESHOLD`, `_CRITICAL_CONF_CAP`), defect names, and the `SHELF_LIFE_TABLE` are strictly hardcoded in the Python files instead of being managed via centralized configurations.
- **Error Handling on URL Fetches**: In `similarity.py`, URL fetching failures are silently skipped with empty bytes passed on, which might quietly distort batch average similarities.

## 🔴 Action Items

- [ ] Implement a pure numpy/Python alternative for Non-Maximum Suppression (NMS) in `yolo_detector.py` to remove the hard `opencv-python` dependency.
- [ ] Migrate `SHELF_LIFE_TABLE` and global vision thresholds (like `_CONF_THRESHOLD`, `_IOU_THRESHOLD` and `GRADE_LABELS`) to a managed config file (e.g., `yaml` or `json`).
- [ ] Enhance telemetry and error reporting in `compare_url_batches` so that missing URLs or network timeouts are surfaced properly instead of silently failing to a neutral score.
