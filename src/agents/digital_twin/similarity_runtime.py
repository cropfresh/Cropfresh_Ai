"""Helpers for loading and evaluating ResNet similarity models."""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from src.agents.quality_assessment.training.model_contracts import (
    ModelContractError,
    load_validated_onnx_session,
    validate_resnet_similarity_session,
)

RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]
EMBED_DIM = 128
SUBSTITUTION_THRESHOLD = 0.50
URL_FETCH_TIMEOUT = 5


def load_similarity_session(model_path: Path):
    """Load a validated ResNet similarity session or return None."""
    if not model_path.exists():
        logger.warning("ResNet50 similarity model not found at {}; using phash fallback", model_path)
        return None
    try:
        return load_validated_onnx_session(model_path, validate_resnet_similarity_session)
    except ModelContractError as err:
        logger.warning("Invalid ResNet50 similarity model contract: {}", err)
        return None
    except Exception as err:  # noqa: BLE001
        logger.warning("Failed loading ResNet50 model: {}", err)
        return None


def preprocess_image(image_bytes: bytes, size: int = 224) -> np.ndarray:
    """Decode bytes to an ImageNet-normalized ResNet tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(RESNET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(RESNET_STD, dtype=np.float32)[:, None, None]
    arr = (arr.transpose(2, 0, 1) - mean) / std
    return arr[np.newaxis, :].astype(np.float32, copy=False)


def phash_similarity(img_a: bytes, img_b: bytes) -> float:
    """Fallback similarity based on perceptual hash distance."""
    try:
        import imagehash  # noqa: PLC0415

        h1 = imagehash.phash(Image.open(io.BytesIO(img_a)))
        h2 = imagehash.phash(Image.open(io.BytesIO(img_b)))
        return round(1.0 - (h1 - h2) / 64.0, 4)
    except Exception as exc:  # noqa: BLE001
        logger.debug("phash similarity failed: {}; returning neutral 0.70", exc)
        return 0.70


def build_batch_result(scores: list[float]) -> dict:
    """Aggregate pairwise similarity scores into the runtime result contract."""
    return {
        "similarity_score": round(float(np.mean(scores)), 4),
        "min_score": round(float(np.min(scores)), 4),
        "substitution_flag": float(np.min(scores)) < SUBSTITUTION_THRESHOLD,
    }


def fetch_url_list(urls: list[str]) -> list[bytes]:
    """Fetch raw bytes for HTTP(S) image URLs, skipping invalid entries."""
    result: list[bytes] = []
    for url in urls:
        if not url.startswith(("http://", "https://")):
            continue
        try:
            with urllib.request.urlopen(url, timeout=URL_FETCH_TIMEOUT) as response:
                result.append(response.read())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to fetch image {}: {}", url, exc)
    return result
