"""Pipeline-level tests for DINOv2 integration with CropVisionPipeline."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


def _make_fake_image() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (120, 120), color=(120, 200, 80)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_logits(best_idx: int, strength: float = 3.0) -> np.ndarray:
    logits = np.zeros(4, dtype=np.float32)
    logits[best_idx] = strength
    return logits


def _mocked_classifier(logits: np.ndarray):
    from src.agents.quality_assessment.dinov2_classifier import DinoV2GradeClassifier

    mock_session = MagicMock()
    mock_session.run.return_value = [logits[np.newaxis, :]]
    clf = DinoV2GradeClassifier.__new__(DinoV2GradeClassifier)
    clf._session = mock_session
    return clf


class TestFullPipelineWithDino:
    @pytest.mark.asyncio
    async def test_vision_mode_uses_dino_grade(self):
        from src.agents.quality_assessment.vision_models import CropVisionPipeline
        from src.agents.quality_assessment.yolo_detector import YoloDefectDetector

        mock_yolo_sess = MagicMock()
        mock_yolo_sess.run.return_value = [np.zeros((1, 1, 14), dtype=np.float32)]
        yolo = YoloDefectDetector.__new__(YoloDefectDetector)
        yolo._session = mock_yolo_sess
        pipeline = CropVisionPipeline.__new__(CropVisionPipeline)
        pipeline.model_dir = "models/vision/"
        pipeline.defect_detector = yolo
        pipeline.grade_classifier = _mocked_classifier(_make_logits(1))
        pipeline.fallback_mode = False

        result = await pipeline.assess_quality(_make_fake_image(), "tomato")

        assert result.assessment_mode == "vision"
        assert result.grade == "A"
        assert 0.0 < result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_pipeline_falls_back_when_dino_unavailable(self):
        from src.agents.quality_assessment.vision_models import CropVisionPipeline
        from src.agents.quality_assessment.dinov2_classifier import GRADE_LABELS

        pipeline = CropVisionPipeline(model_dir="non-existent-dir")
        result = await pipeline.assess_quality(_make_fake_image(), "tomato", "fresh and firm produce")
        assert pipeline.fallback_mode is True
        assert result.assessment_mode == "rule_based"
        assert result.grade in GRADE_LABELS
