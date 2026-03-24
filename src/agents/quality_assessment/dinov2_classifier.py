"""DINOv2 grade classification with commodity-aware ONNX contract validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from src.agents.quality_assessment.dinov2_runtime import (
    build_commodity_tensor,
    preprocess_image,
    softmax,
)
from src.agents.quality_assessment.training.commodity_registry import get_commodity_id
from src.agents.quality_assessment.training.model_contracts import (
    ModelContractError,
    load_validated_onnx_session,
    validate_dino_grade_session,
)

DINO_MODEL_FILENAME = "dinov2_grade_classifier.onnx"
GRADE_LABELS: list[str] = ["A+", "A", "B", "C"]
_INPUT_SIZE = 224
_CRITICAL_DEFECTS: frozenset[str] = frozenset({"rot_spot", "fungal_growth", "overripe"})
_CRITICAL_DOWNGRADE_CAP = "B"
_CRITICAL_CONF_CAP = 0.72
_DEFECT_COUNT_A_PLUS_CAP = 3
_DEFECT_COUNT_A_CAP_GRADE = "A"
_DEFECT_COUNT_A_CAP_CONF = 0.68


def _preprocess(image_bytes: bytes, size: int = _INPUT_SIZE) -> np.ndarray:
    """Backwards-compatible preprocessing alias used by tests."""
    return preprocess_image(image_bytes, size)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Backwards-compatible softmax alias used by tests."""
    return softmax(logits)


def apply_yolo_ensemble(
    raw_grade: str,
    raw_confidence: float,
    detected_defects: list[str],
    confidence_vector: list[float] | None = None,
) -> tuple[str, float, list[float]]:
    """Downgrade optimistic DINO grades when YOLO sees severe defect evidence."""
    critical_found = _CRITICAL_DEFECTS & set(detected_defects)
    vec = confidence_vector or []
    if critical_found and raw_grade in ("A+", "A"):
        logger.debug(
            "Ensemble override: {} found -> downgrade {} -> {}",
            critical_found,
            raw_grade,
            _CRITICAL_DOWNGRADE_CAP,
        )
        return _CRITICAL_DOWNGRADE_CAP, min(raw_confidence, _CRITICAL_CONF_CAP), vec
    if len(detected_defects) > _DEFECT_COUNT_A_PLUS_CAP and raw_grade == "A+":
        logger.debug("Ensemble override: {} defects -> downgrade A+ -> A", len(detected_defects))
        return _DEFECT_COUNT_A_CAP_GRADE, min(raw_confidence, _DEFECT_COUNT_A_CAP_CONF), vec
    return raw_grade, raw_confidence, vec


class DinoV2GradeClassifier:
    """Validated ONNX runtime wrapper for commodity-conditioned grade logits."""

    def __init__(self, model_dir: str = "models/vision/") -> None:
        self._session = self._load_session(Path(model_dir) / DINO_MODEL_FILENAME)

    def _load_session(self, model_path: Path):
        if not model_path.exists():
            logger.warning("DINOv2 grade classifier not found at {}; grade stub will be used", model_path)
            return None
        try:
            session = load_validated_onnx_session(model_path, validate_dino_grade_session)
            logger.info("DINOv2 grade classifier loaded from {}", model_path)
            return session
        except ModelContractError as err:
            logger.warning("Invalid DINOv2 model contract at {}: {}; grade stub active", model_path, err)
            return None
        except Exception as err:  # noqa: BLE001
            logger.warning("Failed loading DINOv2 model: {}; grade stub active", err)
            return None

    @property
    def is_available(self) -> bool:
        return self._session is not None

    def classify(
        self,
        image_bytes: bytes,
        commodity: str = "",
        detected_defects: list[str] | None = None,
    ) -> tuple[str, float, list[float]]:
        """Classify produce grade from image bytes plus a commodity identifier."""
        defects = detected_defects or []
        if not self.is_available:
            logger.warning("DinoV2GradeClassifier.classify() called without model; returning stub")
            return _grade_from_defect_count(len(defects))
        try:
            raw_grade, raw_confidence, prob_vector = self._run_inference(image_bytes, commodity)
            return apply_yolo_ensemble(raw_grade, raw_confidence, defects, prob_vector)
        except Exception as err:  # noqa: BLE001
            logger.warning("DINOv2 inference failed: {}; using defect-count fallback", err)
            return _grade_from_defect_count(len(defects))

    def _run_inference(self, image_bytes: bytes, commodity: str) -> tuple[str, float, list[float]]:
        tensor = preprocess_image(image_bytes, _INPUT_SIZE)
        commodity_id = build_commodity_tensor(get_commodity_id(commodity))
        logits: np.ndarray = self._session.run(
            None,
            {"pixel_values": tensor, "commodity_id": commodity_id},
        )[0][0]
        probs = softmax(logits)
        grade_idx = int(probs.argmax())
        return GRADE_LABELS[grade_idx], float(probs[grade_idx]), [round(float(p), 6) for p in probs]


def _grade_from_defect_count(defect_count: int) -> tuple[str, float, list[float]]:
    """Conservative heuristic used only when the real DINO model is unavailable."""
    if defect_count == 0:
        return "A+", 0.86, [0.86, 0.09, 0.03, 0.02]
    if defect_count <= 2:
        return "A", 0.79, [0.04, 0.79, 0.13, 0.04]
    if defect_count <= 4:
        return "B", 0.68, [0.02, 0.11, 0.68, 0.19]
    return "C", 0.62, [0.01, 0.06, 0.31, 0.62]
