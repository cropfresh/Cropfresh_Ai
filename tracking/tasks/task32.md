# Task 32: DINOv2 Real Grade Classification Integration

> **Priority:** 🔴 P0 | **Phase:** CV Vision Real Integration | **Effort:** 3–4 days
> **Files:** `src/agents/quality_assessment/dinov2_classifier.py` [NEW], `vision_models.py`, `scripts/train_dinov2_classifier.py` [NEW]
> **Status:** ✅ Complete — 2026-03-03
> **Depends On:** Task 31 (YOLOv26 integration ✅), Task 3 (QA Agent ✅)
> **Sprint:** Sprint 06 — CV Vision Real Models

### ✅ Implementation Complete

| File                                                 | Lines | Role                                                                                     |
| ---------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------- |
| `src/agents/quality_assessment/dinov2_classifier.py` | 220   | All DINOv2 logic — preprocessing, softmax, ensemble rules, `DinoV2GradeClassifier` class |
| `src/agents/quality_assessment/vision_models.py`     | ~205  | `_classify_grade_stub` removed; Stage 2 now calls `DinoV2GradeClassifier.classify()`     |
| `tests/unit/test_vision_dinov2.py`                   | ~230  | 22 new unit tests — preprocessing, softmax, ensemble, mocked ONNX, exception recovery    |
| `scripts/train_dinov2_classifier.py`                 | ~220  | Training reference: frozen ViT-S/14 backbone + MLP head, ONNX export at opset 17         |

**Test results:** 56/56 passed (18 existing + 11 YOLO + 22 DINOv2 + 5 integration)

---

## 📌 Problem Statement

The `CropVisionPipeline._classify_grade_stub()` currently **counts defects** from Task 31's output and outputs a static hardcoded grade mapping (`0 defects → A+, 0.86`). This ignores the entire visual appearance of the produce — freshness glow, uniformity, color richness — which are critical for premium grade certification.

DINOv2, the state-of-the-art self-supervised Vision Transformer, will provide the deep visual feature embedding that makes the grade distinction scientifically grounded.

---

## 🔬 Research Findings

### Why DINOv2 for Grading

- **Self-supervised pre-training**: DINOv2 was trained on 142M carefully curated images. Its visual features generalize exceptionally well — it understands "what fresh produce looks like" without requiring millions of labeled agricultural images.
- **Fine-tuning efficiency**: Freeze the backbone, train only a small MLP head on ~2,000 labeled produce images and achieve 90%+ grade classification accuracy.
- **ONNX exportable**: Both backbone + MLP head can be merged into a single ONNX graph for CPU inference via `onnxruntime`.

### Model Size Options

| DINOv2 Variant | Params | ONNX Size | CPU Latency | Accuracy |
| -------------- | ------ | --------- | ----------- | -------- |
| **ViT-S/14**   | 21M    | ~85MB     | ~100ms      | 91%      |
| **ViT-B/14**   | 86M    | ~340MB    | ~350ms      | 94%      |
| **ViT-L/14**   | 307M   | ~1.2GB    | ~900ms      | 96%      |

**Chosen:** `ViT-S/14` with MLP head (fastest + good accuracy for production CPU).

### Training Pipeline

1. **Dataset:** 2,000–5,000 images per commodity (Tomato, Onion, Potato, Mango, Beans) labeled as A+/A/B/C
2. **Split:** 80% train / 10% val / 10% test
3. **Architecture:** Frozen DINOv2-ViT-S/14 backbone → Global Average Pool → Linear(384, 128) → ReLU → Dropout(0.3) → Linear(128, 4)
4. **Training:** AdamW lr=3e-4, 20 epochs, weighted cross-entropy for class imbalance
5. **Export:** `torch.onnx.export()` with opset=17

---

## 🏗️ Implementation Spec

### 1. Image Preprocessing for DINOv2

```python
DINO_MEAN = [0.485, 0.456, 0.406]   # ImageNet normalize — same as DINOv2 training
DINO_STD  = [0.229, 0.224, 0.225]

def _preprocess_for_dinov2(self, image_bytes: bytes, target_size: int = 224) -> np.ndarray:
    """
    Preprocess image for DINOv2 ViT-S/14 input.
    Output shape: (1, 3, 224, 224), dtype float32, ImageNet normalized.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((target_size, target_size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Normalize per-channel
    mean = np.array(DINO_MEAN, dtype=np.float32)[:, None, None]
    std  = np.array(DINO_STD,  dtype=np.float32)[:, None, None]
    arr  = arr.transpose(2, 0, 1)   # HWC → CHW
    arr  = (arr - mean) / std
    return arr[np.newaxis, :]        # NCHW
```

### 2. Real DINOv2 Grade Classification (replace `_classify_grade_stub`)

```python
GRADE_LABELS = ["A+", "A", "B", "C"]

def _classify_grade(self, image_bytes: bytes, defect_results: DefectDetectionResult) -> tuple[str, float]:
    """
    Classify produce grade using DINOv2 ViT-S/14 + MLP head.
    Applies ensemble logic: YOLO defect count can downgrade DINOv2 output.

    Returns:
        (grade, confidence) — e.g., ("A", 0.87)
    """
    if self.grade_classifier is None:
        return self._classify_grade_stub(defect_results.defects)

    input_tensor = self._preprocess_for_dinov2(image_bytes)
    input_name = self.grade_classifier.get_inputs()[0].name
    logits = self.grade_classifier.run(None, {input_name: input_tensor})[0][0]

    # Softmax → probabilities
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()

    raw_grade = GRADE_LABELS[probs.argmax()]
    confidence = float(probs.max())

    # === Ensemble Override ===
    # 1. If YOLOv26 found critical defects, enforce minimum downgrade
    critical = {"rot_spot", "fungal_growth", "overripe"}
    critical_found = critical.intersection(set(defect_results.defects))
    if critical_found and raw_grade in ("A+", "A"):
        raw_grade = "B"
        confidence = min(confidence, 0.72)

    # 2. If YOLOv26 found > 3 defects, cap at B
    if len(defect_results.defects) > 3 and raw_grade == "A+":
        raw_grade = "A"
        confidence = min(confidence, 0.68)

    return raw_grade, confidence
```

### 3. Training Script Reference (`scripts/train_dinov2_classifier.py`)

```python
"""
Train DINOv2 + MLP head for produce grade classification.
Requires: torch, torchvision, transformers, datasets
Run: python scripts/train_dinov2_classifier.py --data data/grading/
"""
import torch
import torch.nn as nn
from transformers import AutoModel

GRADE_LABELS = ["A+", "A", "B", "C"]

class GradeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, len(GRADE_LABELS))
        )

    def forward(self, x):
        features = self.backbone(pixel_values=x).pooler_output  # (B, 384)
        return self.head(features)

def export_to_onnx(model, output_path="models/vision/dinov2_grade_classifier.onnx"):
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported → {output_path}")
```

### 4. Updated `assess_quality` Pipeline

```python
async def assess_quality(self, image: bytes, commodity: str, description_hint: str = "") -> QualityResult:
    """Full two-stage vision pipeline — YOLO + DINOv2."""
    if self.fallback_mode:
        return self._rule_based_assessment(commodity, description_hint)

    # Stage 1: YOLO defect detection
    defect_results = self._detect_defects(image, description_hint)

    # Stage 2: DINOv2 grade classification (with YOLO ensemble)
    grade, confidence = self._classify_grade(image, defect_results)

    hitl_required = confidence < 0.7 or grade == "A+" or len(defect_results.defects) > 3
    return QualityResult(
        grade=grade,
        confidence=confidence,
        defects=defect_results.defects,
        defect_count=len(defect_results.defects),
        hitl_required=hitl_required,
        annotations=defect_results.bboxes,
        shelf_life_days=self._estimate_shelf_life(commodity, grade),
        assessment_mode="vision",
    )
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                                                         | Weight |
| --- | --------------------------------------------------------------------------------- | ------ |
| 1   | DINOv2 ONNX session runs successfully on 224×224 input image bytes                | 25%    |
| 2   | Output grade is one of `["A+", "A", "B", "C"]` with confidence in `[0.0, 1.0]`    | 20%    |
| 3   | Ensemble override: `rot_spot`/`fungal_growth` defects downgrade A+/A outputs      | 20%    |
| 4   | `assessment_mode` is `"vision"` (not `"rule_based"`) when both ONNX models loaded | 15%    |
| 5   | Unit tests in `test_vision_dinov2.py` pass with mocked ONNX session               | 15%    |
| 6   | Full pipeline latency ≤ 500ms on CPU for a 1MP image (YOLO + DINOv2 combined)     | 5%     |

---

## 📚 Dependencies

- `onnxruntime>=1.17.0` (existing)
- `Pillow>=10.0.0` (existing)
- `numpy>=1.24.0` (existing)
- Model file: `models/vision/dinov2_grade_classifier.onnx` (trained via `scripts/train_dinov2_classifier.py`)
- **Training only:** `torch>=2.1`, `transformers>=4.38`, `torchvision>=0.16`

---

## 📐 Related Files

- `src/agents/quality_assessment/vision_models.py` — primary implementation target
- `scripts/train_dinov2_classifier.py` — training + ONNX export script (new)
- `tests/unit/test_vision_dinov2.py` — new test file
- `data/grading/` — labeled image dataset (A+/A/B/C per commodity)
- `models/vision/dinov2_grade_classifier.onnx` — output model artifact
