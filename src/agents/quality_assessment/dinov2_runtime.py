"""Runtime helpers for DINOv2 preprocessing and commodity inputs."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_bytes: bytes, size: int) -> np.ndarray:
    """Decode image bytes to an ImageNet-normalized NCHW tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - _IMAGENET_MEAN[:, None, None]) / _IMAGENET_STD[:, None, None]
    return arr[np.newaxis, :]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for ONNX logits."""
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def build_commodity_tensor(commodity_id: int) -> np.ndarray:
    """Encode a commodity id as the batch-shaped int64 tensor expected by ONNX."""
    return np.array([commodity_id], dtype=np.int64)
