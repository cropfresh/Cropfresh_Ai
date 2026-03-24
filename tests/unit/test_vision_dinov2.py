"""Unit tests for DinoV2GradeClassifier helpers and inference."""

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


def _make_fake_image(w: int = 120, h: int = 120) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (w, h), color=(120, 200, 80)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_logits(best_idx: int, strength: float = 3.0) -> np.ndarray:
    logits = np.zeros(len(GRADE_LABELS), dtype=np.float32)
    logits[best_idx] = strength
    return logits


def _mocked_classifier(logits: np.ndarray) -> DinoV2GradeClassifier:
    mock_session = MagicMock()
    pixel_values = MagicMock()
    pixel_values.name = "pixel_values"
    commodity_id = MagicMock()
    commodity_id.name = "commodity_id"
    mock_session.get_inputs.return_value = [pixel_values, commodity_id]
    mock_session.run.return_value = [logits[np.newaxis, :]]
    clf = DinoV2GradeClassifier.__new__(DinoV2GradeClassifier)
    clf._session = mock_session
    return clf


class TestPreprocess:
    def test_output_shape(self):
        assert _preprocess(_make_fake_image()).shape == (1, 3, 224, 224)

    def test_dtype_float32(self):
        assert _preprocess(_make_fake_image()).dtype == np.float32

    def test_imagenet_normalised(self):
        tensor = _preprocess(_make_fake_image())
        assert tensor.min() < 0.0 or tensor.max() > 1.0


class TestSoftmax:
    def test_output_sums_to_one(self):
        assert abs(_softmax(np.array([1.0, 2.0, 0.5, -0.3], dtype=np.float32)).sum() - 1.0) < 1e-6

    def test_all_values_positive(self):
        assert all(p > 0 for p in _softmax(np.array([-10.0, -5.0, 0.0, 5.0], dtype=np.float32)))

    def test_largest_logit_has_highest_prob(self):
        assert _softmax(np.array([0.1, 0.2, 5.0, 0.3], dtype=np.float32)).argmax() == 2


class TestGradeFromDefectCount:
    @pytest.mark.parametrize(("defect_count", "expected"), [(0, "A+"), (2, "A"), (4, "B"), (5, "C")])
    def test_grade_mapping(self, defect_count: int, expected: str):
        assert _grade_from_defect_count(defect_count)[0] == expected

    def test_confidence_is_in_range(self):
        for defect_count in range(6):
            _, confidence, _ = _grade_from_defect_count(defect_count)
            assert 0.0 < confidence <= 1.0


class TestApplyYoloEnsemble:
    def test_critical_defect_downgrades_a_plus(self):
        assert apply_yolo_ensemble("A+", 0.91, ["rot_spot"])[0] == "B"

    def test_critical_defect_downgrades_a(self):
        assert apply_yolo_ensemble("A", 0.85, ["fungal_growth"])[0] == "B"

    def test_critical_defect_caps_confidence(self):
        assert apply_yolo_ensemble("A+", 0.95, ["overripe"])[1] <= 0.72

    def test_critical_defect_does_not_downgrade_b(self):
        grade, confidence, _ = apply_yolo_ensemble("B", 0.75, ["rot_spot"])
        assert grade == "B"
        assert confidence == pytest.approx(0.75)

    def test_too_many_defects_downgrades_a_plus(self):
        assert apply_yolo_ensemble("A+", 0.88, ["bruise", "bruise", "colour_off", "surface_crack"])[0] == "A"

    def test_many_defects_do_not_affect_a(self):
        grade, confidence, _ = apply_yolo_ensemble("A", 0.78, ["bruise"] * 5)
        assert grade == "A"
        assert confidence == pytest.approx(0.78)

    def test_no_defects_no_change(self):
        grade, confidence, _ = apply_yolo_ensemble("A+", 0.92, [])
        assert grade == "A+"
        assert confidence == pytest.approx(0.92)


class TestDinoV2GradeClassifier:
    def test_is_available_false_when_model_missing(self):
        assert DinoV2GradeClassifier(model_dir="non-existent-dir").is_available is False

    def test_classify_falls_back_when_model_missing(self):
        grade, confidence, _ = DinoV2GradeClassifier(model_dir="non-existent-dir").classify(_make_fake_image(), detected_defects=[])
        assert grade in GRADE_LABELS
        assert 0.0 < confidence <= 1.0

    def test_classify_a_plus_with_mocked_session(self):
        clf = _mocked_classifier(_make_logits(0))
        grade, confidence, _ = clf.classify(_make_fake_image(), commodity="tomato", detected_defects=[])
        assert grade == "A+"
        assert confidence > 0.7
        _, feed_dict = clf._session.run.call_args.args
        assert feed_dict["commodity_id"].tolist() == [1]

    def test_classify_c_with_mocked_session(self):
        assert _mocked_classifier(_make_logits(3)).classify(_make_fake_image(), detected_defects=[])[0] == "C"

    def test_ensemble_applied_on_mocked_inference(self):
        grade, confidence, _ = _mocked_classifier(_make_logits(0)).classify(_make_fake_image(), detected_defects=["rot_spot"])
        assert grade == "B"
        assert confidence <= 0.72

    def test_inference_exception_returns_fallback(self):
        mock_session = MagicMock()
        mock_session.run.side_effect = RuntimeError("ONNX crash")
        clf = DinoV2GradeClassifier.__new__(DinoV2GradeClassifier)
        clf._session = mock_session
        grade, confidence, _ = clf.classify(_make_fake_image(), detected_defects=["bruise", "colour_off"])
        assert grade == "A"
        assert 0.0 < confidence <= 1.0

    def test_confidence_is_probability(self):
        _, confidence, _ = _mocked_classifier(_make_logits(1)).classify(_make_fake_image())
        assert 0.0 < confidence <= 1.0
