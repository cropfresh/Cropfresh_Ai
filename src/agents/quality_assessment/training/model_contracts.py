"""ONNX model-contract validation for CropFresh vision models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

EXPECTED_GRADE_COUNT = 4
EXPECTED_DEFECT_CLASS_COUNT = 10


class ModelContractError(ValueError):
    """Raised when a model file does not match the expected runtime contract."""


def load_validated_onnx_session(
    model_path: Path,
    validator: Callable[[Any], None],
    providers: list[str] | None = None,
):
    """Load an ONNX session and validate its public tensor contract."""
    import onnxruntime as ort  # noqa: PLC0415

    session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
    validator(session)
    return session


def validate_yolo_session(session: Any) -> None:
    """Require a YOLO output shape that matches CropFresh defect classes only."""
    output = session.get_outputs()[0]
    shape = tuple(output.shape)
    expected_channel_width = EXPECTED_DEFECT_CLASS_COUNT + 4
    valid = len(shape) == 3 and expected_channel_width in shape[1:]
    if not valid:
        raise ModelContractError(
            f"YOLO contract mismatch: expected output with 4+{EXPECTED_DEFECT_CLASS_COUNT} channels, got {shape}"
        )


def validate_dino_grade_session(session: Any) -> None:
    """Require commodity-conditioned grade logits from the DINO classifier."""
    input_names = {tensor.name for tensor in session.get_inputs()}
    output = session.get_outputs()[0]
    shape = tuple(output.shape)
    if {"pixel_values", "commodity_id"} - input_names:
        raise ModelContractError(
            f"DINO contract mismatch: expected inputs pixel_values + commodity_id, got {input_names}"
        )
    if output.name != "logits" or len(shape) < 2 or shape[-1] != EXPECTED_GRADE_COUNT:
        raise ModelContractError(
            f"DINO contract mismatch: expected logits(batch,{EXPECTED_GRADE_COUNT}), got {output.name}{shape}"
        )


def validate_resnet_similarity_session(session: Any) -> None:
    """Require a 128-d embedding output for similarity inference."""
    output = session.get_outputs()[0]
    shape = tuple(output.shape)
    if output.name != "embedding" or len(shape) < 2 or shape[-1] != 128:
        raise ModelContractError(
            f"ResNet contract mismatch: expected embedding(batch,128), got {output.name}{shape}"
        )
