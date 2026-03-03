"""
DINOv2 grade classification module.

Single responsibility: run DINOv2 ViT-S/14 + MLP head ONNX inference to
assign produce quality grades (A+/A/B/C) from produce images.

Also owns the YOLO ensemble override logic — if severe defects are detected
by Stage 1 (yolo_detector.py), this module can downgrade the DINOv2 output
before returning the final grade.

Imported by vision_models.CropVisionPipeline; never call directly from agents.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

# * ─── constants ─────────────────────────────────────────────────────────────

DINO_MODEL_FILENAME = "dinov2_grade_classifier.onnx"

# Grade labels — index order must match the MLP head's output neurons.
GRADE_LABELS: list[str] = ["A+", "A", "B", "C"]

# * ImageNet mean/std — DINOv2 was pre-trained on ImageNet; normalise the same way.
# Using the same statistics keeps the feature space aligned with backbone training.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# DINOv2 ViT-S/14 expects 224×224 tokens — fixed by the patch embedding layer.
_INPUT_SIZE: int = 224

# ! Changing these thresholds will affect HITL trigger rates — tune carefully.
# Critical defects detected by YOLO that warrant enforced grade downgrade.
_CRITICAL_DEFECTS: frozenset[str] = frozenset({"rot_spot", "fungal_growth", "overripe"})

# If DINOv2 says A+ or A but YOLO found one of these, downgrade to B.
_CRITICAL_DOWNGRADE_CAP = "B"
_CRITICAL_CONF_CAP = 0.72

# If YOLO found more than this many defects on an A+ prediction, cap at A.
_DEFECT_COUNT_A_PLUS_CAP = 3
_DEFECT_COUNT_A_CAP_GRADE = "A"
_DEFECT_COUNT_A_CAP_CONF = 0.68


# * ─── preprocessing ─────────────────────────────────────────────────────────


def _preprocess(image_bytes: bytes, size: int = _INPUT_SIZE) -> np.ndarray:
    """
    Decode raw image bytes → normalised NCHW float32 tensor for DINOv2.

    Output shape: (1, 3, 224, 224), dtype float32.
    Bicubic resize preserves colour and sharpness better than bilinear for
    the fine-grained freshness features DINOv2 relies on.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0    # HWC, [0, 1]
    arr = arr.transpose(2, 0, 1)                      # CHW
    mean = _IMAGENET_MEAN[:, None, None]
    std  = _IMAGENET_STD[:, None, None]
    arr  = (arr - mean) / std                         # ImageNet normalised
    return arr[np.newaxis, :]                          # NCHW


# * ─── softmax helper ─────────────────────────────────────────────────────────


def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    Subtracting the max prevents overflow in exp() for large logit values.
    """
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# * ─── ensemble logic ─────────────────────────────────────────────────────────


def apply_yolo_ensemble(
    raw_grade: str,
    raw_confidence: float,
    detected_defects: list[str],
) -> tuple[str, float]:
    """
    Post-process DINOv2 output with YOLO defect evidence.

    Rules (applied in order — first rule that fires wins):
    1. Critical defects (rot_spot, fungal_growth, overripe) found on A+/A → force B
    2. More than 3 defects on an A+ prediction → cap at A

    This keeps the pipeline conservative: visual freshness from DINOv2 can be
    upgraded, but hard physical defects from YOLO can only downgrade.
    """
    defect_set = set(detected_defects)
    critical_found = _CRITICAL_DEFECTS & defect_set

    # Rule 1: critical defect forces downgrade of premium grades
    if critical_found and raw_grade in ("A+", "A"):
        logger.debug(
            "Ensemble override: {} found → downgrade {} → {}",
            critical_found, raw_grade, _CRITICAL_DOWNGRADE_CAP,
        )
        return _CRITICAL_DOWNGRADE_CAP, min(raw_confidence, _CRITICAL_CONF_CAP)

    # Rule 2: too many defects disqualifies A+
    if len(detected_defects) > _DEFECT_COUNT_A_PLUS_CAP and raw_grade == "A+":
        logger.debug(
            "Ensemble override: {} defects → downgrade A+ → A",
            len(detected_defects),
        )
        return _DEFECT_COUNT_A_CAP_GRADE, min(raw_confidence, _DEFECT_COUNT_A_CAP_CONF)

    return raw_grade, raw_confidence


# * ─── main classifier class ──────────────────────────────────────────────────


class DinoV2GradeClassifier:
    """
    Wraps a DINOv2 ViT-S/14 + MLP head ONNX session for produce grade classification.

    Graceful degradation: if the model file is missing or fails to load at
    startup the classifier signals unavailability via `is_available = False` and
    the pipeline falls back to rule-based mode.
    """

    def __init__(self, model_dir: str = "models/vision/") -> None:
        self._session = self._load_session(Path(model_dir) / DINO_MODEL_FILENAME)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _load_session(self, model_path: Path):
        """Return an onnxruntime InferenceSession or None on failure."""
        if not model_path.exists():
            logger.warning(
                "DINOv2 grade classifier not found at {}; grade stub will be used",
                model_path,
            )
            return None
        try:
            import onnxruntime as ort  # noqa: PLC0415

            sess = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("DINOv2 grade classifier loaded from {}", model_path)
            return sess
        except Exception as err:
            logger.warning("Failed loading DINOv2 model: {}; grade stub active", err)
            return None

    @property
    def is_available(self) -> bool:
        return self._session is not None

    # ── public classification API ──────────────────────────────────────────

    def classify(
        self,
        image_bytes: bytes,
        detected_defects: list[str] | None = None,
    ) -> tuple[str, float]:
        """
        Classify produce grade from raw image bytes.

        Args:
            image_bytes: Raw JPEG/PNG/WebP bytes of the produce image.
            detected_defects: Defect labels from Stage 1 YOLO run.
                              Used for ensemble override logic.

        Returns:
            (grade, confidence) — e.g. ("A", 0.87).
            Grade is one of GRADE_LABELS; confidence ∈ (0.0, 1.0).
        """
        defects = detected_defects or []

        if not self.is_available:
            # ? Should never reach here in normal flow — CropVisionPipeline
            # ? checks is_available before calling; kept as safety net.
            logger.warning("DinoV2GradeClassifier.classify() called without model; returning stub")
            return _grade_from_defect_count(len(defects))

        try:
            raw_grade, raw_confidence = self._run_inference(image_bytes)
            return apply_yolo_ensemble(raw_grade, raw_confidence, defects)
        except Exception as err:
            # ! Inference error: fall back to defect-count stub rather than crash.
            logger.warning("DINOv2 inference failed: {}; using defect-count fallback", err)
            return _grade_from_defect_count(len(defects))

    # ── private inference helpers ──────────────────────────────────────────

    def _run_inference(self, image_bytes: bytes) -> tuple[str, float]:
        """Preprocess → forward pass → softmax → (grade, confidence)."""
        tensor = _preprocess(image_bytes)
        input_name = self._session.get_inputs()[0].name
        logits: np.ndarray = self._session.run(None, {input_name: tensor})[0][0]

        probs = _softmax(logits)
        grade_idx = int(probs.argmax())
        return GRADE_LABELS[grade_idx], float(probs[grade_idx])


# * ─── defect-count grade stub (fallback only) ────────────────────────────────


def _grade_from_defect_count(defect_count: int) -> tuple[str, float]:
    """
    Conservative heuristic used ONLY when DINOv2 model is unavailable.
    Kept inside this module so it can be removed entirely in one place
    once all prod environments have the ONNX model deployed.
    """
    if defect_count == 0:
        return "A+", 0.86
    if defect_count <= 2:
        return "A", 0.79
    if defect_count <= 4:
        return "B", 0.68
    return "C", 0.62
