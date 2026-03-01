# Task 3: Implement Quality Assessment Agent (CV-QG Vision Grading)

> **Priority:** 🔴 P0 | **Phase:** 1 | **Effort:** 4–5 days | **Status:** ✅ Completed (2026-03-01)  
> **Files:** `src/agents/quality_assessment/agent.py`, `src/agents/quality_assessment/vision_models.py` [NEW]  
> **Score Target:** 9/10 — >95% grading accuracy, HITL trigger when confidence <0.7

---

## 📌 Problem Statement

The business model requires AI-powered produce grading achieving >95% accuracy by Month 12. Currently the quality assessment agent exists (11KB) but needs:
- Real vision model integration (YOLOv8 for defect detection + DINOv2/ViT for grade classification)
- HITL (Human-in-the-Loop) flag for low-confidence predictions
- Digital Twin linkage for departure/arrival comparison

---

## 🔬 Research Findings

### Vision Model Stack (2025 State-of-the-Art)
| Model | Role | Size | Accuracy | Latency |
|-------|------|------|----------|---------|
| **YOLOv8n** | Defect detection (bbox) | 6.2MB | mAP 85%+ | <50ms CPU |
| **YOLOv12s** | Enhanced defect detection (attention) | 11MB | mAP 89%+ | ~80ms CPU |
| **DINOv2-ViT-S/14** | Grade classification (feature extraction) | 21MB | 92%+ accuracy | ~100ms CPU |
| **MobileNetV3-Small** | Lightweight grade classifier | 2.5MB | 88% accuracy | <30ms CPU |

### Grade Schema (Business-Aligned)
| Grade | Description | Defects Allowed | Price Premium |
|-------|-------------|-----------------|---------------|
| **A+** | Premium — export quality | 0 defects, uniform size+color | +20% above modal |
| **A** | Good — retail quality | ≤2 minor surface marks | +10% above modal |
| **B** | Fair — wholesale quality | ≤5 surface defects, minor color variance | Modal price |
| **C** | Utility — processing quality | Multiple defects, irregular size | -15% below modal |

### Defect Categories
```
bruise, worm_hole, colour_off, size_irregular, surface_crack,
rot_spot, fungal_growth, mechanical_damage, overripe, underripe
```

### HITL (Human-in-the-Loop) Rules
- AI confidence < 0.7 → flag for field agent verification
- Grade upgrade request from farmer → always HITL
- Disputed grade (post-delivery) → HITL + comparison photos
- Category A+ → always requires agent photo verification

---

## 🏗️ Implementation Spec

### 1. Vision Model Wrapper (`vision_models.py`)
```python
class CropVisionPipeline:
    """
    Two-stage vision pipeline for produce quality assessment.
    
    Stage 1: YOLOv8 defect detection → bounding boxes + defect types
    Stage 2: DINOv2 feature extraction → grade classifier
    
    Graceful degradation:
    - No GPU → CPU inference (ONNX Runtime)
    - No model weights → rule-based fallback
    - No image → manual grade entry accepted
    """
    
    def __init__(self, model_dir: str = "models/vision/"):
        self.defect_detector = self._load_yolo(model_dir)
        self.grade_classifier = self._load_classifier(model_dir)
        self.fallback_mode = self.defect_detector is None
    
    def _load_yolo(self, model_dir: str):
        """Load YOLOv8n ONNX model. Returns None if not available."""
        model_path = Path(model_dir) / "yolov8n_agri_defects.onnx"
        if not model_path.exists():
            logger.warning("YOLOv8 model not found, using rule-based fallback")
            return None
        import onnxruntime as ort
        return ort.InferenceSession(str(model_path))
    
    async def assess_quality(
        self,
        image: bytes,
        commodity: str,
    ) -> QualityResult:
        """
        Full quality assessment pipeline.
        
        Returns:
            QualityResult with grade, confidence, defects, and HITL flag
        """
        if self.fallback_mode:
            return self._rule_based_assessment(commodity)
        
        # Stage 1: Defect detection
        preprocessed = self._preprocess(image, target_size=(640, 640))
        defect_results = self._detect_defects(preprocessed)
        
        # Stage 2: Grade classification
        features = self._extract_features(preprocessed)
        grade, confidence = self._classify_grade(features, defect_results)
        
        # HITL decision
        hitl_required = (
            confidence < 0.7 or
            grade == 'A+' or  # Premium always verified
            len(defect_results.defects) > 3  # Many defects = uncertain
        )
        
        return QualityResult(
            grade=grade,
            confidence=confidence,
            defects=defect_results.defects,
            defect_count=len(defect_results.defects),
            hitl_required=hitl_required,
            annotations=defect_results.bboxes,
            shelf_life_days=self._estimate_shelf_life(commodity, grade),
        )
```

### 2. Enhanced Agent (`agent.py`)
```python
class QualityAssessmentAgent(BaseAgent):
    """
    CV-QG Agent — Computer Vision Quality Grading.
    
    Capabilities:
    1. Photo-based produce grading (A+/A/B/C)
    2. Defect detection with bounding boxes
    3. Shelf life prediction
    4. HITL escalation for low confidence
    5. Digital twin creation linkage
    """
    
    async def execute(self, input_data: dict) -> dict:
        image = input_data.get('image')
        commodity = input_data.get('commodity', 'tomato')
        
        if image:
            result = await self.vision_pipeline.assess_quality(image, commodity)
        else:
            # Manual grading via LLM conversation
            result = await self._manual_grade_conversation(input_data)
        
        # Store in DB + link to digital twin
        await self._save_assessment(result, input_data)
        
        return {
            'grade': result.grade,
            'confidence': result.confidence,
            'defects': result.defects,
            'hitl_required': result.hitl_required,
            'shelf_life_days': result.shelf_life_days,
            'message': self._format_result_message(result, commodity),
        }
```

### 3. Shelf Life Predictor
```python
SHELF_LIFE_TABLE = {
    # (commodity, grade): days at room temp
    ('tomato', 'A+'): 7,
    ('tomato', 'A'): 5,
    ('tomato', 'B'): 3,
    ('tomato', 'C'): 1,
    ('onion', 'A+'): 30,
    ('onion', 'A'): 25,
    ('potato', 'A+'): 45,
    ('beans', 'A+'): 3,
    ('mango', 'A+'): 5,
}

def _estimate_shelf_life(self, commodity: str, grade: str) -> int:
    """Estimate shelf life in days at room temperature."""
    key = (commodity.lower(), grade)
    return SHELF_LIFE_TABLE.get(key, 3)  # Default 3 days
```

---

## ✅ Acceptance Criteria (9/10 Score)

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Grades produce into A+/A/B/C with confidence score | 25% |
| 2 | Defect detection returns specific defect types | 20% |
| 3 | HITL flag triggers when confidence < 0.7 | 15% |
| 4 | Graceful fallback when vision model not available | 15% |
| 5 | Shelf life prediction per commodity × grade | 10% |
| 6 | Digital twin linkage (stores assessment ID) | 10% |
| 7 | Unit tests with mock image data pass | 5% |

---

## 📚 Dependencies
- `onnxruntime` — CPU inference for ONNX models
- `Pillow` — image preprocessing
- `numpy` — tensor operations
- Model weights: Download separately (YOLOv8n ~6MB, DINOv2 ~21MB)

---

## ✅ Completion Update (2026-03-01)

### Implemented
- Added `src/agents/quality_assessment/vision_models.py` with `CropVisionPipeline`:
  - Optional ONNX model loading (`yolov8n_agri_defects.onnx`, `dinov2_grade_classifier.onnx`)
  - Graceful fallback mode when model weights are unavailable
  - A+/A/B/C grade output with confidence, defect list, annotations, and shelf-life estimate
- Upgraded `src/agents/quality_assessment/agent.py`:
  - Integrated `CropVisionPipeline` into `assess()`
  - Added `execute(input_data)` contract for orchestrated calls
  - Enforced HITL policy: confidence `< 0.7`, `A+` grade, or explicit upgrade-review request
  - Added digital twin linkage via generated assessment IDs (`qa-...`) and in-memory report store
- Wired quality agent into runtime:
  - Registered `quality_assessment_agent` in both chat bootstraps
  - Added supervisor rule-based routing keywords for quality assessment intents
- Updated static dashboard for sprint visibility:
  - Added quick buyer-matching and quick quality-check widgets in `static/index.html`
  - Added handlers in `static/assets/js/dashboard.js`

### Validation
- `uv run pytest tests/unit/test_quality_assessment.py tests/unit/test_supervisor_routing.py`
- Result: **27 passed**

### Acceptance Criteria Mapping
1. Grades into A+/A/B/C with confidence → ✅  
2. Defect detection returns specific defect types → ✅  
3. HITL triggers for confidence `< 0.7` → ✅  
4. Graceful fallback without vision weights → ✅  
5. Shelf life prediction by commodity × grade → ✅  
6. Digital twin linkage with assessment ID → ✅  
7. Unit tests pass with mock/fallback flows → ✅
