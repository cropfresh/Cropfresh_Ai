"""Unit tests for ResNet similarity contract enforcement at runtime."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.agents.digital_twin.similarity import ResNetSimilarityEngine


def _make_image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=(120, 180, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_engine_marks_placeholder_model_unavailable():
    engine = ResNetSimilarityEngine(model_dir="models/vision/")
    assert engine.available is False


def test_engine_embed_works_with_injected_valid_session():
    class FakeSession:
        def run(self, *_args, **_kwargs):
            return [np.ones((1, 128), dtype=np.float32)]

    engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
    engine.session = FakeSession()
    embedding = engine.embed(_make_image_bytes())
    assert embedding is not None
    assert embedding.shape == (128,)
