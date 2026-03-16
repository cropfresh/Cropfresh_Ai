"""YOLO defect detection runtime for CropFresh quality grading."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from src.agents.quality_assessment.training.model_contracts import (
    ModelContractError,
    load_validated_onnx_session,
    validate_yolo_session,
)

YOLO_MODEL_FILENAME = "yolov26n_agri_defects.onnx"
DEFECT_CLASS_NAMES: list[str] = [
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
_CONF_THRESHOLD = 0.35
_IOU_THRESHOLD = 0.45
_INPUT_SIZE = 640


class DefectBox(BaseModel):
    """Single detected defect with location and score."""

    label: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionResult(BaseModel):
    """Aggregated YOLO detection output."""

    defects: list[str] = Field(default_factory=list)
    boxes: list[DefectBox] = Field(default_factory=list)

    def to_annotation_dicts(self) -> list[dict]:
        return [box.model_dump() for box in self.boxes]


def _preprocess(image_bytes: bytes, size: int = _INPUT_SIZE) -> np.ndarray:
    """Decode raw image bytes into an NCHW float32 tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis, :]


def _apply_nms(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO predictions and apply non-maximum suppression."""
    import cv2  # noqa: PLC0415

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), _CONF_THRESHOLD, _IOU_THRESHOLD)
    if len(keep) == 0:
        empty = np.array([])
        return empty, empty, empty.astype(int)
    keep = keep.flatten()
    return boxes_xyxy[keep], confidences[keep], class_ids[keep]


def _boxes_to_result(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> DetectionResult:
    """Map raw NMS arrays into the typed DetectionResult contract."""
    seen: set[str] = set()
    defect_boxes: list[DefectBox] = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score < _CONF_THRESHOLD:
            continue
        label = DEFECT_CLASS_NAMES[int(class_id)]
        seen.add(label)
        defect_boxes.append(
            DefectBox(
                label=label,
                score=round(float(score), 3),
                x1=int(box[0]),
                y1=int(box[1]),
                x2=int(box[2]),
                y2=int(box[3]),
            )
        )
    return DetectionResult(defects=list(seen), boxes=defect_boxes)


def detect_from_description(description: str) -> DetectionResult:
    """Keyword fallback when no valid YOLO model is available."""
    desc_lower = description.lower()
    found = [label for label in DEFECT_CLASS_NAMES if label in desc_lower or label.replace("_", " ") in desc_lower]
    return DetectionResult(defects=found, boxes=[])


class YoloDefectDetector:
    """Validated ONNX wrapper for CropFresh produce-defect detection."""

    def __init__(self, model_dir: str = "models/vision/") -> None:
        self._session = self._load_session(Path(model_dir) / YOLO_MODEL_FILENAME)

    def _load_session(self, model_path: Path):
        if not model_path.exists():
            logger.warning("YOLOv26 model not found at {}; entering text-fallback mode", model_path)
            return None
        try:
            session = load_validated_onnx_session(model_path, validate_yolo_session)
            logger.info("YOLOv26 defect detector loaded from {}", model_path)
            return session
        except ModelContractError as err:
            logger.warning("Invalid YOLOv26 model contract: {}; text-fallback active", err)
            return None
        except Exception as err:  # noqa: BLE001
            logger.warning("Failed loading YOLOv26 model: {}; text-fallback active", err)
            return None

    @property
    def is_available(self) -> bool:
        return self._session is not None

    def detect(self, image_bytes: bytes, description_hint: str = "") -> DetectionResult:
        """Run ONNX detection or fall back to keyword extraction."""
        if not self.is_available:
            return detect_from_description(description_hint)
        try:
            return self._run_inference(image_bytes)
        except Exception as err:  # noqa: BLE001
            logger.warning("YOLOv26 inference failed: {}; using text fallback", err)
            return detect_from_description(description_hint)

    def _run_inference(self, image_bytes: bytes) -> DetectionResult:
        tensor = _preprocess(image_bytes)
        predictions = self._session.run(None, {"images": tensor})[0][0]
        boxes, scores, class_ids = _apply_nms(predictions)
        if len(boxes) == 0:
            return DetectionResult()
        return _boxes_to_result(boxes, scores, class_ids)
