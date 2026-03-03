"""
Unit tests for YoloDefectDetector (Task 31).

All tests mock the ONNX InferenceSession so no model weights are needed.
Tests cover: successful inference, NMS filtering, fallback modes, and the
text-based keyword extractor.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.quality_assessment.yolo_detector import (
    DEFECT_CLASS_NAMES,
    DetectionResult,
    YoloDefectDetector,
    _apply_nms,
    _preprocess,
    detect_from_description,
)


# * ─── helpers ────────────────────────────────────────────────────────────────

def _make_fake_image(width: int = 100, height: int = 100) -> bytes:
    """Return minimal valid JPEG bytes for preprocessing tests."""
    from io import BytesIO
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(180, 120, 60))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_raw_predictions(
    boxes_xywh: list[list[float]],
    class_scores: list[list[float]],
) -> np.ndarray:
    """
    Build a fake YOLO output tensor matching the expected (N, 4+C) layout.
    boxes_xywh: list of [cx, cy, w, h] in pixel coords
    class_scores: list of per-class confidence vectors (length = num classes)
    """
    n = len(boxes_xywh)
    num_classes = len(DEFECT_CLASS_NAMES)
    pred = np.zeros((n, 4 + num_classes), dtype=np.float32)
    for i, (box, scores) in enumerate(zip(boxes_xywh, class_scores)):
        pred[i, :4] = box
        pred[i, 4:] = scores
    return pred


# * ─── preprocessing ──────────────────────────────────────────────────────────

class TestPreprocess:
    def test_output_shape_is_nchw(self):
        tensor = _preprocess(_make_fake_image())
        assert tensor.shape == (1, 3, 640, 640)

    def test_values_are_normalised(self):
        tensor = _preprocess(_make_fake_image())
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_dtype_is_float32(self):
        tensor = _preprocess(_make_fake_image())
        assert tensor.dtype == np.float32


# * ─── text fallback ──────────────────────────────────────────────────────────

class TestDetectFromDescription:
    def test_known_keyword_detected(self):
        result = detect_from_description("some bruise and rot_spot visible")
        assert "bruise" in result.defects
        assert "rot_spot" in result.defects

    def test_phrase_with_space_detected(self):
        # "worm hole" (space) must map to "worm_hole" (underscore class name)
        result = detect_from_description("has worm hole damage")
        assert "worm_hole" in result.defects

    def test_no_match_returns_empty(self):
        result = detect_from_description("premium fresh uniform tomatoes")
        assert result.defects == []

    def test_always_returns_empty_boxes(self):
        # Text fallback can never produce spatial bounding boxes
        result = detect_from_description("bruise and overripe")
        assert result.boxes == []


# * ─── apply NMS ──────────────────────────────────────────────────────────────

class TestApplyNms:
    def test_empty_predictions_return_empty_arrays(self):
        preds = np.zeros((0, 4 + len(DEFECT_CLASS_NAMES)), dtype=np.float32)
        boxes, scores, ids = _apply_nms(preds)
        assert len(boxes) == 0 and len(scores) == 0 and len(ids) == 0

    def test_high_confidence_box_is_kept(self):
        scores = [0.0] * len(DEFECT_CLASS_NAMES)
        scores[0] = 0.9  # bruise, high confidence
        preds = _make_raw_predictions([[320, 320, 64, 64]], [scores])
        boxes, confs, ids = _apply_nms(preds)
        assert len(boxes) == 1
        assert confs[0] > 0.35

    def test_low_confidence_box_is_filtered(self):
        scores = [0.0] * len(DEFECT_CLASS_NAMES)
        scores[1] = 0.10  # worm_hole, below threshold
        preds = _make_raw_predictions([[100, 100, 40, 40]], [scores])
        boxes, confs, ids = _apply_nms(preds)
        assert len(boxes) == 0


# * ─── YoloDefectDetector ─────────────────────────────────────────────────────

class TestYoloDefectDetector:

    def test_is_available_false_when_model_missing(self):
        detector = YoloDefectDetector(model_dir="non-existent-dir")
        assert detector.is_available is False

    def test_detect_falls_back_to_description_when_unavailable(self):
        detector = YoloDefectDetector(model_dir="non-existent-dir")
        result = detector.detect(_make_fake_image(), "bruise visible")
        # Text fallback must return bruise but no spatial boxes
        assert "bruise" in result.defects
        assert result.boxes == []

    def test_detect_with_mocked_session(self):
        """Verify full inference path with a mocked ONNX session."""
        # Build a fake one-detection output tensor
        scores = [0.0] * len(DEFECT_CLASS_NAMES)
        scores[5] = 0.85   # rot_spot at high confidence
        fake_preds = _make_raw_predictions([[300, 300, 80, 80]], [scores])
        fake_output = [fake_preds[np.newaxis, :]]  # (1, N, 4+C)

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.run.return_value = fake_output

        detector = YoloDefectDetector.__new__(YoloDefectDetector)
        detector._session = mock_session

        result = detector.detect(_make_fake_image())

        assert "rot_spot" in result.defects
        assert len(result.boxes) >= 1
        assert result.boxes[0].label == "rot_spot"
        assert result.boxes[0].score == pytest.approx(0.85, abs=0.01)

    def test_detect_returns_empty_result_on_no_detections(self):
        """All-zero prediction tensor → empty result, no crash."""
        num_classes = len(DEFECT_CLASS_NAMES)
        fake_preds = np.zeros((1, 1, 4 + num_classes), dtype=np.float32)

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.run.return_value = [fake_preds]

        detector = YoloDefectDetector.__new__(YoloDefectDetector)
        detector._session = mock_session

        result = detector.detect(_make_fake_image())
        assert isinstance(result, DetectionResult)
        assert result.defects == []

    def test_inference_exception_triggers_text_fallback(self):
        """An unexpected runtime error in the ONNX session must not propagate."""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.run.side_effect = RuntimeError("ONNX runtime crash")

        detector = YoloDefectDetector.__new__(YoloDefectDetector)
        detector._session = mock_session

        # Should not raise — gracefully returns text fallback result
        result = detector.detect(_make_fake_image(), "bruise")
        assert "bruise" in result.defects


# * ─── integration: pipeline fallback mode ────────────────────────────────────

class TestPipelineFallback:

    @pytest.mark.asyncio
    async def test_assess_quality_in_fallback_mode_uses_rule_based(self):
        from src.agents.quality_assessment.vision_models import CropVisionPipeline

        pipeline = CropVisionPipeline(model_dir="non-existent-dir")
        assert pipeline.fallback_mode is True

        result = await pipeline.assess_quality(
            _make_fake_image(), "tomato", "fresh and firm"
        )
        assert result.assessment_mode == "rule_based"
        assert result.grade in ("A+", "A", "B", "C")

    @pytest.mark.asyncio
    async def test_assess_quality_vision_mode_uses_yolo(self):
        """With mocked ONNX running for both stages, assessment_mode must be 'vision'."""
        from src.agents.quality_assessment.dinov2_classifier import DinoV2GradeClassifier
        from src.agents.quality_assessment.vision_models import CropVisionPipeline
        from src.agents.quality_assessment.yolo_detector import YoloDefectDetector

        # ── Stage 1: YOLO with one rot_spot detection ──────────────────────
        scores = [0.0] * len(DEFECT_CLASS_NAMES)
        scores[5] = 0.80  # rot_spot
        fake_preds = _make_raw_predictions([[320, 300, 60, 60]], [scores])

        mock_yolo_sess = MagicMock()
        mock_yolo_sess.get_inputs.return_value = [MagicMock(name="images")]
        mock_yolo_sess.run.return_value = [fake_preds[np.newaxis, :]]

        yolo = YoloDefectDetector.__new__(YoloDefectDetector)
        yolo._session = mock_yolo_sess

        # ── Stage 2: DINOv2 predicting grade "A" (index 1) ────────────────
        dino_logits = np.array([0.1, 3.5, 0.2, 0.1], dtype=np.float32)  # "A" dominant
        mock_dino_sess = MagicMock()
        mock_dino_sess.get_inputs.return_value = [MagicMock(name="pixel_values")]
        mock_dino_sess.run.return_value = [dino_logits[np.newaxis, :]]

        dino = DinoV2GradeClassifier.__new__(DinoV2GradeClassifier)
        dino._session = mock_dino_sess

        # ── Assemble pipeline ──────────────────────────────────────────────
        pipeline = CropVisionPipeline.__new__(CropVisionPipeline)
        pipeline.model_dir = "models/vision/"
        pipeline.defect_detector = yolo
        pipeline.grade_classifier = dino
        pipeline.fallback_mode = False

        result = await pipeline.assess_quality(_make_fake_image(), "tomato")

        assert result.assessment_mode == "vision"
        # rot_spot was detected by YOLO; ensemble may override grade — just check it's valid
        assert result.grade in ("A+", "A", "B", "C")
        assert "rot_spot" in result.defects
        assert len(result.annotations) >= 1

