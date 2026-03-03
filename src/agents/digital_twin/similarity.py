"""
ResNet50 Image Similarity Engine — Digital Twin Verification
=============================================================
Computes cosine similarity between departure and arrival produce images
using a ResNet50 backbone fine-tuned with contrastive/triplet loss.

Primary entry points:
    engine.similarity(img_a, img_b)          → float [0.0, 1.0]
    engine.compare_batches(dep, arr)          → dict
    engine.compare_url_batches(dep_urls, ...) → dict
"""

# * SIMILARITY ENGINE MODULE
# NOTE: ONNX model is optional; degrades to perceptual hash then neutral fallback.
# NOTE: compare_url_batches() mirrors the URL-fetching strategy in diff_analysis.py.
# ! Model file path is relative to the CWD at runtime — ensure models/vision/ is accessible.

from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

# * ImageNet normalisation constants for ResNet preprocessing
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD  = [0.229, 0.224, 0.225]

# * L2-normalised embedding dimension produced by the fine-tuned head
EMBED_DIM = 128

# * Threshold below which min pairwise score triggers substitution alert
SUBSTITUTION_THRESHOLD = 0.50

# * URL fetch timeout — matches diff_analysis.IMAGE_FETCH_TIMEOUT
URL_FETCH_TIMEOUT = 5


# * ═══════════════════════════════════════════════════════════════
# * ResNet Similarity Engine
# * ═══════════════════════════════════════════════════════════════

class ResNetSimilarityEngine:
    """
    Visual similarity between two produce images using a ResNet50 backbone
    fine-tuned with contrastive loss and a 128-dim L2-normalised head.

    Graceful degradation order:
        1. ResNet50 ONNX embedding + cosine similarity  (model available)
        2. Perceptual hash (pHash) Hamming distance      (imagehash installed)
        3. Neutral score 0.70                            (all else fails)

    Usage:
        engine = ResNetSimilarityEngine()
        score = engine.similarity(img_bytes_a, img_bytes_b)
        batch = engine.compare_url_batches(departure_urls, arrival_urls)
    """

    def __init__(self, model_dir: str = "models/vision/") -> None:
        model_path = Path(model_dir) / "resnet50_produce_similarity.onnx"
        self.session = self._load_session(model_path)

    # ─────────────────────────────────────────────────────────
    # * Public — availability probe
    # ─────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when ONNX model is loaded and ready for inference."""
        return self.session is not None

    # ─────────────────────────────────────────────────────────
    # * Public — bytes-level API (raw image bytes)
    # ─────────────────────────────────────────────────────────

    def embed(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract 128-dim L2-normalised embedding for a single image.

        Returns:
            np.ndarray of shape (128,) or None when model unavailable.
        """
        if self.session is None:
            return None
        tensor = self._preprocess(image_bytes)
        input_name = self.session.get_inputs()[0].name
        raw = self.session.run(None, {input_name: tensor})[0][0]   # (128,)
        norm = np.linalg.norm(raw)
        return raw / norm if norm > 0 else raw

    def similarity(self, img_a: bytes, img_b: bytes) -> float:
        """
        Cosine similarity ∈ [0.0, 1.0] between two produce images.
        Falls back to perceptual hash when model is unavailable.
        """
        emb_a = self.embed(img_a)
        emb_b = self.embed(img_b)
        if emb_a is None or emb_b is None:
            return self._phash_similarity(img_a, img_b)
        return float(np.clip(np.dot(emb_a, emb_b), 0.0, 1.0))

    def compare_batches(
        self,
        departure_images: list[bytes],
        arrival_images:   list[bytes],
    ) -> dict:
        """
        Compare entire departure vs arrival batch (bytes-level).

        Samples up to 3 images from each side and computes all pairwise
        similarities, returning the average and minimum scores.

        Returns:
            dict with keys:
                similarity_score (float): Mean pairwise cosine similarity.
                min_score        (float): Minimum pairwise score.
                substitution_flag (bool): True when min_score < 0.50.
        """
        if not departure_images or not arrival_images:
            return {"similarity_score": 0.5, "min_score": 0.5, "substitution_flag": False}

        scores = [
            self.similarity(dep, arr)
            for dep in departure_images[:3]
            for arr in arrival_images[:3]
        ]
        return _build_batch_result(scores)

    # ─────────────────────────────────────────────────────────
    # * Public — URL-level API (S3 / HTTP URLs)
    # ─────────────────────────────────────────────────────────

    def compare_url_batches(
        self,
        departure_urls: list[str],
        arrival_urls:   list[str],
    ) -> dict:
        """
        Compare departure vs arrival batches from HTTP(S) URL strings.

        Samples up to 3 URLs from each side. Non-HTTP(S) URLs (e.g. s3://)
        are silently skipped — if no valid URLs are loadable the method
        returns a neutral result (similarity_score=0.70, no substitution flag).

        Returns:
            Same dict shape as compare_batches().
        """
        dep_bytes = _fetch_url_list(departure_urls[:3])
        arr_bytes = _fetch_url_list(arrival_urls[:3])

        if not dep_bytes or not arr_bytes:
            # ? No fetchable photos — return neutral score, no false alarm
            logger.debug("compare_url_batches: no HTTP photos available, returning neutral score")
            return {"similarity_score": 0.70, "min_score": 0.70, "substitution_flag": False}

        return self.compare_batches(dep_bytes, arr_bytes)

    # ─────────────────────────────────────────────────────────
    # * Private — ONNX session loader
    # ─────────────────────────────────────────────────────────

    def _load_session(self, model_path: Path):
        """Load ONNX InferenceSession; returns None on any failure."""
        if not model_path.exists():
            logger.warning(
                "ResNet50 similarity model not found at {}; using phash fallback",
                model_path,
            )
            return None
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_path))
            logger.info("ResNet50 similarity engine loaded from {}", model_path)
            return session
        except Exception as err:
            logger.warning("Failed loading ResNet50 model: {}", err)
            return None

    # ─────────────────────────────────────────────────────────
    # * Private — image preprocessing
    # ─────────────────────────────────────────────────────────

    def _preprocess(self, image_bytes: bytes, size: int = 224) -> np.ndarray:
        """Decode bytes → CHW float32 tensor normalised for ResNet, shape (1, 3, 224, 224)."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((size, size), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array(RESNET_MEAN, dtype=np.float32)[:, None, None]
        std  = np.array(RESNET_STD,  dtype=np.float32)[:, None, None]
        arr  = (arr.transpose(2, 0, 1) - mean) / std
        return arr[np.newaxis, :]   # (1, 3, H, W)

    # ─────────────────────────────────────────────────────────
    # * Private — perceptual hash fallback
    # ─────────────────────────────────────────────────────────

    def _phash_similarity(self, img_a: bytes, img_b: bytes) -> float:
        """
        Perceptual hash similarity when model is unavailable.
        Hamming distance D ∈ [0, 64] → similarity = 1 - D/64.
        """
        try:
            import imagehash
            h1 = imagehash.phash(Image.open(io.BytesIO(img_a)))
            h2 = imagehash.phash(Image.open(io.BytesIO(img_b)))
            distance = h1 - h2      # Hamming distance [0, 64]
            return round(1.0 - distance / 64.0, 4)
        except Exception as exc:
            logger.debug("phash similarity failed: {}; returning neutral 0.70", exc)
            return 0.70             # Neutral fallback


# * ═══════════════════════════════════════════════════════════════
# * Module-level pure helpers (stateless, testable)
# * ═══════════════════════════════════════════════════════════════

def _build_batch_result(scores: list[float]) -> dict:
    """Aggregate a flat list of pairwise scores into the standard batch result dict."""
    avg_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    return {
        "similarity_score": round(avg_score, 4),
        "min_score":        round(min_score, 4),
        "substitution_flag": min_score < SUBSTITUTION_THRESHOLD,
    }


def _fetch_url_list(urls: list[str]) -> list[bytes]:
    """
    Fetch raw bytes for a list of HTTP(S) URLs.
    Non-HTTP URLs and failed fetches are silently skipped.
    """
    result: list[bytes] = []
    for url in urls:
        if not url.startswith(("http://", "https://")):
            continue
        try:
            with urllib.request.urlopen(url, timeout=URL_FETCH_TIMEOUT) as resp:
                result.append(resp.read())
        except Exception as exc:
            logger.debug("Failed to fetch image {}: {}", url, exc)
    return result
