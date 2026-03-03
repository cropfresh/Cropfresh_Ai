"""
Unit tests for DinoV2GradeClassifier (Task 32).

All tests mock the ONNX InferenceSession — no model weights required.
Tests cover: preprocessing, softmax, YOLO ensemble rules, mocked inference,
fallback behaviour, and pipeline integration.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.agents.quality_assessment.dinov2_classifier import (
    GRADE_LABELS,
    DinoV2GradeClassifier,
    _grade_from_defect_count,
    _preprocess,
    _softmax,
    apply_yolo_ensemble,
)


# * ─── helpers ────────────────────────────────────────────────────────────────

def _make_fake_image(w: int = 120, h: int = 120) -> bytes:
    """Return minimal valid JPEG bytes for preprocessing tests."""
    img = Image.new("RGB", (w, h), color=(120, 200, 80))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_logits(best_idx: int, strength: float = 3.0) -> np.ndarray:
    """Build a logit vector where the given class clearly dominates."""
    logits = np.zeros(len(GRADE_LABELS), dtype=np.float32)
    logits[best_idx] = strength
    return logits


def _mocked_classifier(logits: np.ndarray) -> DinoV2GradeClassifier:
    """Return a DinoV2GradeClassifier with a mocked ONNX session."""
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="pixel_values")]
    # ONNX output: [(batch, num_classes)] — use shape (1, 4)
    mock_session.run.return_value = [logits[np.newaxis, :]]

    clf = DinoV2GradeClassifier.__new__(DinoV2GradeClassifier)
    clf._session = mock_session
    return clf


# * ─── preprocessing tests ────────────────────────────────────────────────────

class TestPreprocess:
    def test_output_shape(self):
        tensor = _preprocess(_make_fake_image())
        assert tensor.shape == (1, 3, 224, 224)

    def test_dtype_float32(self):
        assert _preprocess(_make_fake_image()).dtype == np.float32

    def test_imagenet_normalised(self):
        # After ImageNet normalisation, pixel values extend beyond [0, 1].
        # We check that normalisation actually happened (not raw [0,1] range).
        tensor = _preprocess(_make_fake_image())
        # Pure green image gets normalised differently per channel — just check
        # the full tensor is not already clipped to [0, 1].
        assert tensor.min() < 0.0 or tensor.max() > 1.0


# * ─── softmax tests ──────────────────────────────────────────────────────────

class TestSoftmax:
    def test_output_sums_to_one(self):
        logits = np.array([1.0, 2.0, 0.5, -0.3], dtype=np.float32)
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_all_values_positive(self):
        logits = np.array([-10.0, -5.0, 0.0, 5.0], dtype=np.float32)
        probs = _softmax(logits)
        assert all(p > 0 for p in probs)

    def test_largest_logit_has_highest_prob(self):
        logits = np.array([0.1, 0.2, 5.0, 0.3], dtype=np.float32)
        assert _softmax(logits).argmax() == 2


# * ─── defect-count fallback ──────────────────────────────────────────────────

class TestGradeFromDefectCount:
    def test_zero_defects_is_a_plus(self):
        assert _grade_from_defect_count(0)[0] == "A+"

    def test_two_defects_is_a(self):
        assert _grade_from_defect_count(2)[0] == "A"

    def test_four_defects_is_b(self):
        assert _grade_from_defect_count(4)[0] == "B"

    def test_five_defects_is_c(self):
        assert _grade_from_defect_count(5)[0] == "C"

    def test_confidence_is_in_range(self):
        for n in range(6):
            _, conf = _grade_from_defect_count(n)
            assert 0.0 < conf <= 1.0


# * ─── YOLO ensemble override ─────────────────────────────────────────────────

class TestApplyYoloEnsemble:
    def test_critical_defect_downgrades_a_plus(self):
        grade, _ = apply_yolo_ensemble("A+", 0.91, ["rot_spot"])
        assert grade == "B"

    def test_critical_defect_downgrades_a(self):
        grade, _ = apply_yolo_ensemble("A", 0.85, ["fungal_growth"])
        assert grade == "B"

    def test_critical_defect_caps_confidence(self):
        _, conf = apply_yolo_ensemble("A+", 0.95, ["overripe"])
        assert conf <= 0.72

    def test_critical_defect_does_not_downgrade_b(self):
        # B is already below the premium tier — no further forced downgrade
        grade, conf = apply_yolo_ensemble("B", 0.75, ["rot_spot"])
        assert grade == "B"
        assert conf == pytest.approx(0.75)

    def test_too_many_defects_downgrades_a_plus(self):
        grade, _ = apply_yolo_ensemble("A+", 0.88, ["bruise", "bruise", "colour_off", "surface_crack"])
        assert grade == "A"

    def test_many_defects_do_not_affect_a(self):
        # Rule 2 only targets A+; A should be untouched
        grade, conf = apply_yolo_ensemble("A", 0.78, ["bruise"] * 5)
        assert grade == "A"
        assert conf == pytest.approx(0.78)

    def test_no_defects_no_change(self):
        grade, conf = apply_yolo_ensemble("A+", 0.92, [])
        assert grade == "A+"
        assert conf == pytest.approx(0.92)


# * ─── DinoV2GradeClassifier ─────────────────────────────────────────────────

class TestDinoV2GradeClassifier:

    def test_is_available_false_when_model_missing(self):
        clf = DinoV2GradeClassifier(model_dir="non-existent-dir")
        assert clf.is_available is False

    def test_classify_falls_back_when_model_missing(self):
        clf = DinoV2GradeClassifier(model_dir="non-existent-dir")
        grade, conf = clf.classify(_make_fake_image(), detected_defects=[])
        assert grade in GRADE_LABELS
        assert 0.0 < conf <= 1.0

    def test_classify_a_plus_with_mocked_session(self):
        """High A+ logit → should return A+."""
        clf = _mocked_classifier(_make_logits(0))   # index 0 = "A+"
        grade, conf = clf.classify(_make_fake_image(), detected_defects=[])
        assert grade == "A+"
        assert conf > 0.7

    def test_classify_c_with_mocked_session(self):
        clf = _mocked_classifier(_make_logits(3))   # index 3 = "C"
        grade, conf = clf.classify(_make_fake_image(), detected_defects=[])
        assert grade == "C"

    def test_ensemble_applied_on_mocked_inference(self):
        """A+ logit + rot_spot → ensemble should downgrade to B."""
        clf = _mocked_classifier(_make_logits(0))   # "A+" logit
        grade, conf = clf.classify(_make_fake_image(), detected_defects=["rot_spot"])
        assert grade == "B"
        assert conf <= 0.72

    def test_inference_exception_returns_fallback(self):
        """RuntimeError from ONNX must not propagate; defect-count fallback used."""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="pixel_values")]
        mock_session.run.side_effect = RuntimeError("ONNX crash")

        clf = DinoV2GradeClassifier.__new__(DinoV2GradeClassifier)
        clf._session = mock_session

        grade, conf = clf.classify(_make_fake_image(), detected_defects=["bruise", "colour_off"])
        # 2 defects → _grade_from_defect_count returns "A"
        assert grade == "A"
        assert 0.0 < conf <= 1.0

    def test_confidence_is_probability(self):
        """Returned confidence must be a valid probability."""
        clf = _mocked_classifier(_make_logits(1))
        _, conf = clf.classify(_make_fake_image())
        assert 0.0 < conf <= 1.0


# * ─── integration: full pipeline with both stages ────────────────────────────

class TestFullPipelineWithDino:

    @pytest.mark.asyncio
    async def test_vision_mode_uses_dino_grade(self):
        """With both YOLO and DINOv2 mocked, assessment_mode must be 'vision'."""
        from src.agents.quality_assessment.vision_models import CropVisionPipeline
        from src.agents.quality_assessment.yolo_detector import YoloDefectDetector

        # Mock YOLO: no defects detected
        mock_yolo_sess = MagicMock()
        mock_yolo_sess.get_inputs.return_value = [MagicMock(name="images")]
        import numpy as _np
        num_cls = 10
        mock_yolo_sess.run.return_value = [
            _np.zeros((1, 1, 4 + num_cls), dtype=_np.float32)
        ]
        yolo = YoloDefectDetector.__new__(YoloDefectDetector)
        yolo._session = mock_yolo_sess

        # Mock DINOv2: predicts "A" (index 1)
        dino = _mocked_classifier(_make_logits(1))

        pipeline = CropVisionPipeline.__new__(CropVisionPipeline)
        pipeline.model_dir = "models/vision/"
        pipeline.defect_detector = yolo
        pipeline.grade_classifier = dino
        pipeline.fallback_mode = False

        result = await pipeline.assess_quality(_make_fake_image(), "tomato")

        assert result.assessment_mode == "vision"
        assert result.grade == "A"
        assert 0.0 < result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_pipeline_falls_back_when_dino_unavailable(self):
        """If DINOv2 model is missing, full pipeline must use rule_based mode."""
        from src.agents.quality_assessment.vision_models import CropVisionPipeline

        pipeline = CropVisionPipeline(model_dir="non-existent-dir")
        assert pipeline.fallback_mode is True

        result = await pipeline.assess_quality(
            _make_fake_image(), "tomato", "fresh and firm produce"
        )
        assert result.assessment_mode == "rule_based"
        assert result.grade in GRADE_LABELS
