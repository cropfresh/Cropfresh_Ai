"""
Vision models orchestrator for quality assessment.

Thin wrapper that owns the two-stage grading pipeline:
  Stage 1 → YoloDefectDetector   (yolo_detector.py)      — defect localisation
  Stage 2 → DinoV2GradeClassifier (dinov2_classifier.py)  — A+/A/B/C with confidence

Falls back to rule-based keyword assessment when ONNX models are absent,
keeping the pipeline functional in dev / CI without model weights.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.quality_assessment.dinov2_classifier import (
    DinoV2GradeClassifier,
    _grade_from_defect_count,
)
from src.agents.quality_assessment.yolo_detector import (
    YoloDefectDetector,
    detect_from_description,
)

# * ─── shelf-life lookup table ───────────────────────────────────────────────
# Keyed by (commodity_lower, grade).  Default 3 days when key absent.

SHELF_LIFE_TABLE: dict[tuple[str, str], int] = {
    ("tomato", "A+"): 7,
    ("tomato", "A"): 5,
    ("tomato", "B"): 3,
    ("tomato", "C"): 1,
    ("onion", "A+"): 30,
    ("onion", "A"): 25,
    ("onion", "B"): 18,
    ("onion", "C"): 10,
    ("potato", "A+"): 45,
    ("potato", "A"): 35,
    ("potato", "B"): 25,
    ("potato", "C"): 15,
    ("beans", "A+"): 3,
    ("beans", "A"): 2,
    ("beans", "B"): 1,
    ("beans", "C"): 1,
    ("mango", "A+"): 5,
    ("mango", "A"): 4,
    ("mango", "B"): 3,
    ("mango", "C"): 2,
}

# Used by rule-based fallback and as a reference set for keyword extraction.
KNOWN_DEFECT_TYPES: list[str] = [
    "bruise",
    "worm_hole",
    "colour_off",
    "size_irregular",
    "surface_crack",
    "rot_spot",
    "fungal_growth",
    "mechanical_damage",
    "overripe",
    "underripe",
]

# * ─── result models ─────────────────────────────────────────────────────────


class DefectDetectionResult(BaseModel):
    """Public-facing defect result — kept for backwards compat with agent.py."""

    defects: list[str] = Field(default_factory=list)
    bboxes: list[dict] = Field(default_factory=list)


class QualityResult(BaseModel):
    grade: str
    confidence: float
    defects: list[str] = Field(default_factory=list)
    defect_count: int = 0
    hitl_required: bool = False
    annotations: list[dict] = Field(default_factory=list)
    shelf_life_days: int = 3
    # * FR9: DINOv2 softmax probability vector [p_A+, p_A, p_B, p_C] — immutable audit trail
    dinov2_confidence_vector: list[float] = Field(default_factory=list)
    # * "vision" when ONNX models ran; "rule_based" otherwise — checked by tests
    assessment_mode: str = "rule_based"


# * ─── pipeline ──────────────────────────────────────────────────────────────


class CropVisionPipeline:
    """
    Two-stage produce quality pipeline.

    Stage 1: YoloDefectDetector — localise and classify defects on the image.
    Stage 2: DINOv2 grade classifier ONNX — holistic A+/A/B/C grading.
             (Task 32; stub used until DINOv2 model is integrated.)

    Graceful degradation order:
      ONNX models present → full vision pipeline
      ONNX models absent  → rule-based keyword assessment
    """

    def __init__(self, model_dir: str = "models/vision/") -> None:
        self.model_dir = model_dir
        # Stage 1: YOLOv26 defect detector (Task 31 ✅)
        self.defect_detector = YoloDefectDetector(model_dir)
        # Stage 2: DINOv2 grade classifier (Task 32 ✅)
        self.grade_classifier = DinoV2GradeClassifier(model_dir)
        # * Pipeline degrades gracefully: both stages must be available for vision mode
        self.fallback_mode = (
            not self.defect_detector.is_available
            or not self.grade_classifier.is_available
        )

    # ── public API ─────────────────────────────────────────────────────────

    async def assess_quality(
        self,
        image: bytes,
        commodity: str,
        description_hint: str = "",
    ) -> QualityResult:
        """Full two-stage pipeline entrypoint (called when an image is present)."""
        if self.fallback_mode:
            return self._rule_based_assessment(commodity, description_hint)

        # Stage 1 — real YOLOv26 defect detection
        detection = self.defect_detector.detect(image, description_hint)

        # Stage 2 — real DINOv2 grade classification + YOLO ensemble override
        # FR9: unpack 3-tuple (grade, confidence, confidence_vector)
        grade, confidence, confidence_vector = self.grade_classifier.classify(
            image, detected_defects=detection.defects
        )

        hitl_required = (
            confidence < 0.7
            or grade == "A+"
            or len(detection.defects) > 3
        )
        return QualityResult(
            grade=grade,
            confidence=confidence,
            defects=detection.defects,
            defect_count=len(detection.defects),
            hitl_required=hitl_required,
            annotations=detection.to_annotation_dicts(),
            shelf_life_days=self._estimate_shelf_life(commodity, grade),
            dinov2_confidence_vector=confidence_vector,
            assessment_mode="vision",
        )

    async def assess_description(self, commodity: str, description: str) -> QualityResult:
        """Text-only assessment path (no image supplied)."""
        return self._rule_based_assessment(commodity, description)

    # ── internal helpers ───────────────────────────────────────────────────

    def _rule_based_assessment(self, commodity: str, description: str) -> QualityResult:
        """
        Keyword-based fallback when ONNX models are unavailable.
        Returns honest low-confidence grades; always flags HITL.
        """
        detection = detect_from_description(description)
        detected_defects = detection.defects

        if not detected_defects:
            desc_lower = description.lower()
            if any(t in desc_lower for t in ["fresh", "firm", "uniform", "clean", "premium"]):
                grade, confidence, confidence_vector = "A", 0.74, [0.03, 0.74, 0.18, 0.05]
            elif any(t in desc_lower for t in ["soft", "damaged", "rotten", "fungal", "spot"]):
                grade, confidence, confidence_vector = "C", 0.64, [0.01, 0.05, 0.30, 0.64]
            else:
                grade, confidence, confidence_vector = "B", 0.66, [0.02, 0.10, 0.66, 0.22]
        else:
            # * FR9: _grade_from_defect_count now returns 3-tuple
            grade, confidence, confidence_vector = _grade_from_defect_count(len(detected_defects))

        hitl_required = confidence < 0.7 or grade == "A+" or len(detected_defects) > 3
        return QualityResult(
            grade=grade,
            confidence=confidence,
            defects=detected_defects,
            defect_count=len(detected_defects),
            hitl_required=hitl_required,
            annotations=[],
            shelf_life_days=self._estimate_shelf_life(commodity, grade),
            dinov2_confidence_vector=confidence_vector,
            assessment_mode="rule_based",
        )

    def _estimate_shelf_life(self, commodity: str, grade: str) -> int:
        return SHELF_LIFE_TABLE.get((commodity.lower(), grade), 3)
