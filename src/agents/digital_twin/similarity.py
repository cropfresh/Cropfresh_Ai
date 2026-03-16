"""ResNet-backed similarity engine for Digital Twin verification."""

from __future__ import annotations

import numpy as np

from src.agents.digital_twin.similarity_runtime import (
    EMBED_DIM,
    SUBSTITUTION_THRESHOLD,
    build_batch_result,
    fetch_url_list,
    load_similarity_session,
    phash_similarity,
    preprocess_image,
)

_build_batch_result = build_batch_result
_fetch_url_list = fetch_url_list


class ResNetSimilarityEngine:
    """Compare departure and arrival produce photos using validated embeddings."""

    def __init__(self, model_dir: str = "models/vision/") -> None:
        from pathlib import Path

        self.session = load_similarity_session(Path(model_dir) / "resnet50_produce_similarity.onnx")

    @property
    def available(self) -> bool:
        return self.session is not None

    def embed(self, image_bytes: bytes) -> np.ndarray | None:
        """Return a normalized embedding or None when the model is unavailable."""
        if self.session is None:
            return None
        raw = self.session.run(None, {"image": self._preprocess(image_bytes)})[0][0]
        norm = np.linalg.norm(raw)
        return raw / norm if norm > 0 else raw

    def similarity(self, img_a: bytes, img_b: bytes) -> float:
        """Cosine similarity clipped to [0, 1] with pHash fallback."""
        emb_a = self.embed(img_a)
        emb_b = self.embed(img_b)
        if emb_a is None or emb_b is None:
            return self._phash_similarity(img_a, img_b)
        return float(np.clip(np.dot(emb_a, emb_b), 0.0, 1.0))

    def compare_batches(self, departure_images: list[bytes], arrival_images: list[bytes]) -> dict:
        """Average pairwise similarity across up to 3 departure and 3 arrival images."""
        if not departure_images or not arrival_images:
            return {"similarity_score": 0.5, "min_score": 0.5, "substitution_flag": False}
        scores = [
            self.similarity(dep, arr)
            for dep in departure_images[:3]
            for arr in arrival_images[:3]
        ]
        return _build_batch_result(scores)

    def compare_url_batches(self, departure_urls: list[str], arrival_urls: list[str]) -> dict:
        """Fetch HTTP(S) photos, then compare them using the batch API."""
        dep_bytes = _fetch_url_list(departure_urls[:3])
        arr_bytes = _fetch_url_list(arrival_urls[:3])
        if not dep_bytes or not arr_bytes:
            return {"similarity_score": 0.70, "min_score": 0.70, "substitution_flag": False}
        return self.compare_batches(dep_bytes, arr_bytes)

    def _preprocess(self, image_bytes: bytes, size: int = 224) -> np.ndarray:
        """Backwards-compatible preprocessing hook used by existing tests."""
        return preprocess_image(image_bytes, size=size)

    def _phash_similarity(self, img_a: bytes, img_b: bytes) -> float:
        """Backwards-compatible pHash hook used by existing tests."""
        return phash_similarity(img_a, img_b)
