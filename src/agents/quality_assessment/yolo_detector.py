"""
YOLOv26 defect detection module.

Single responsibility: run YOLOv26n ONNX inference on produce images
and return bounding-box annotations for each detected defect.

Imported by vision_models.CropVisionPipeline; never call directly from agents.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

# * ─── constants ────────────────────────────────────────────────────────────

YOLO_MODEL_FILENAME = "yolov26n_agri_defects.onnx"

# Order must match the class indices used during YOLO training.
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

# ! Tune these thresholds against your validation set before going to prod.
_CONF_THRESHOLD: float = 0.35
_IOU_THRESHOLD: float = 0.45

# YOLOv26 nano uses 640×640 input — do NOT change without re-exporting the model.
_INPUT_SIZE: int = 640


# * ─── result model ─────────────────────────────────────────────────────────


class DefectBox(BaseModel):
    """A single detected defect with its localisation bbox."""

    label: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionResult(BaseModel):
    """Aggregated output from one inference pass."""

    defects: list[str] = Field(default_factory=list)
    boxes: list[DefectBox] = Field(default_factory=list)

    def to_annotation_dicts(self) -> list[dict]:
        """Convert to the flat dict format expected by QualityResult."""
        return [b.model_dump() for b in self.boxes]


# * ─── preprocessing ────────────────────────────────────────────────────────


def _preprocess(image_bytes: bytes, size: int = _INPUT_SIZE) -> np.ndarray:
    """
    Decode raw image bytes → normalised NCHW float32 tensor suitable for
    YOLOv26 ONNX input.

    Output shape: (1, 3, size, size), dtype float32, range [0, 1].
    Bilinear resize avoids aliasing on produce textures (bruise edges matter).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC
    arr = arr.transpose(2, 0, 1)                    # CHW
    return arr[np.newaxis, :]                        # NCHW


# * ─── post-processing ──────────────────────────────────────────────────────


def _apply_nms(
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode raw YOLOv26 output tensor and apply Non-Maximum Suppression.

    YOLOv26 ONNX output: (num_anchors, 4 + num_classes), cx/cy/w/h format.
    Returns (boxes_xyxy, confidences, class_ids) after NMS filtering.
    """
    # ? cv2.dnn.NMSBoxes is the most reliable CPU NMS available without torch.
    # If opencv-headless ever causes issues we can swap in a pure-numpy impl.
    import cv2  # noqa: PLC0415 — lazy import; never installed at module load

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Convert centre-format → corner-format for NMS
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        _CONF_THRESHOLD,
        _IOU_THRESHOLD,
    )
    if len(keep) == 0:
        empty = np.array([])
        return empty, empty, empty.astype(int)

    keep = keep.flatten()
    return boxes_xyxy[keep], confidences[keep], class_ids[keep]


def _boxes_to_result(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> DetectionResult:
    """Map raw NMS output arrays into a typed DetectionResult."""
    seen_defects: set[str] = set()
    defect_boxes: list[DefectBox] = []

    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < _CONF_THRESHOLD:
            continue
        label = DEFECT_CLASS_NAMES[int(cls_id)]
        seen_defects.add(label)
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

    return DetectionResult(defects=list(seen_defects), boxes=defect_boxes)


# * ─── text-based fallback ──────────────────────────────────────────────────


def detect_from_description(description: str) -> DetectionResult:
    """
    Keyword-based defect extraction used when no image or ONNX model is
    available.  This is an honest fallback — it never produces bounding boxes.
    """
    desc_lower = description.lower()
    found = [
        d for d in DEFECT_CLASS_NAMES
        if d in desc_lower or d.replace("_", " ") in desc_lower
    ]
    return DetectionResult(defects=found, boxes=[])


# * ─── main detector class ───────────────────────────────────────────────────


class YoloDefectDetector:
    """
    Wraps a YOLOv26n ONNX session for produce defect detection.

    Gracefully degrades: if the model file is missing at startup the detector
    falls back to keyword matching via detect_from_description().
    """

    def __init__(self, model_dir: str = "models/vision/") -> None:
        self._session = self._load_session(Path(model_dir) / YOLO_MODEL_FILENAME)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _load_session(self, model_path: Path):
        """Return an onnxruntime InferenceSession or None on failure."""
        if not model_path.exists():
            logger.warning(
                "YOLOv26 model not found at {}; entering text-fallback mode",
                model_path,
            )
            return None
        try:
            import onnxruntime as ort  # noqa: PLC0415

            sess = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("YOLOv26 defect detector loaded from {}", model_path)
            return sess
        except Exception as err:
            logger.warning("Failed loading YOLOv26 model: {}; text-fallback active", err)
            return None

    @property
    def is_available(self) -> bool:
        return self._session is not None

    # ── public detection API ───────────────────────────────────────────────

    def detect(
        self,
        image_bytes: bytes,
        description_hint: str = "",
    ) -> DetectionResult:
        """
        Run defect detection on raw image bytes.

        Falls back to description keyword matching when the ONNX session is
        unavailable — keeps the pipeline functional end-to-end at all times.
        """
        if not self.is_available:
            return detect_from_description(description_hint)

        try:
            return self._run_inference(image_bytes)
        except Exception as err:
            # ! Inference error — fall back silently rather than crashing the
            # ! assessment pipeline; the agent will still return a rule-based grade.
            logger.warning("YOLOv26 inference failed: {}; using text fallback", err)
            return detect_from_description(description_hint)

    # ── private inference helpers ──────────────────────────────────────────

    def _run_inference(self, image_bytes: bytes) -> DetectionResult:
        """Preprocess → forward pass → NMS → typed result."""
        tensor = _preprocess(image_bytes)
        input_name = self._session.get_inputs()[0].name
        raw_output = self._session.run(None, {input_name: tensor})

        # YOLOv26 ONNX: output[0] shape (1, num_anchors, 4+classes)
        predictions = raw_output[0][0]
        boxes, scores, class_ids = _apply_nms(predictions)

        if len(boxes) == 0:
            return DetectionResult()

        return _boxes_to_result(boxes, scores, class_ids)
