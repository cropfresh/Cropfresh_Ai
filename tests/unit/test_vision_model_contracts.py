"""Unit tests for ONNX contract validation of CropFresh vision models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import onnxruntime as ort
import pytest

from src.agents.quality_assessment.training.model_contracts import (
    ModelContractError,
    validate_dino_grade_session,
    validate_resnet_similarity_session,
    validate_yolo_session,
)


def _mock_session(inputs: list[tuple[str, tuple]], outputs: list[tuple[str, tuple]]):
    return SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name=name, shape=shape) for name, shape in inputs],
        get_outputs=lambda: [SimpleNamespace(name=name, shape=shape) for name, shape in outputs],
    )


def test_rejects_current_placeholder_yolo_model():
    session = ort.InferenceSession(str(Path("models/vision/yolov26n_agri_defects.onnx")), providers=["CPUExecutionProvider"])
    with pytest.raises(ModelContractError, match="YOLO contract mismatch"):
        validate_yolo_session(session)


def test_rejects_current_placeholder_dino_model():
    session = ort.InferenceSession(str(Path("models/vision/dinov2_grade_classifier.onnx")), providers=["CPUExecutionProvider"])
    with pytest.raises(ModelContractError, match="DINO contract mismatch"):
        validate_dino_grade_session(session)


def test_rejects_current_placeholder_resnet_model():
    session = ort.InferenceSession(str(Path("models/vision/resnet50_produce_similarity.onnx")), providers=["CPUExecutionProvider"])
    with pytest.raises(ModelContractError, match="ResNet contract mismatch"):
        validate_resnet_similarity_session(session)


def test_accepts_valid_mocked_contracts():
    validate_yolo_session(_mock_session([("images", (1, 3, 640, 640))], [("output0", (1, 8400, 14))]))
    validate_dino_grade_session(_mock_session([("pixel_values", ("batch", 3, 224, 224)), ("commodity_id", ("batch",))], [("logits", ("batch", 4))]))
    validate_resnet_similarity_session(_mock_session([("image", ("batch", 3, 224, 224))], [("embedding", ("batch", 128))]))
