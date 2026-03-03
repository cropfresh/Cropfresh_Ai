# Task 33: ResNet50 Digital Twin Image Similarity Engine

> **Priority:** 🔴 P0 | **Phase:** CV Vision Real Integration | **Effort:** 2–3 days
> **Files:** `src/agents/digital_twin/similarity.py` [NEW], `src/agents/digital_twin/engine.py`, `src/agents/digital_twin/liability.py`, `scripts/train_resnet_similarity.py` [NEW]
> **Status:** [x] Complete — 2026-03-03
> **Depends On:** Task 10 (Digital Twin Engine ✅), Task 31 (YOLO ✅), Task 32 (DINOv2 ✅)
> **Sprint:** Sprint 05 — CV Vision Real Models

---

## 📌 Problem Statement

The `DigitalTwinEngine` currently uses **rule-based transit time degradation** to estimate arrival grade — there is zero actual visual comparison between departure and arrival photos. The `compute_similarity` helper returns a fixed heuristic score. This means the Digital Twin **cannot fulfill its core promise** of legally defensible dispute resolution.

ResNet50 (fine-tuned for contrastive embedding similarity) will transform the Digital Twin into a true image forensics system by comparing the departure batch photos against the arrival photos mathematically.

---

## 🔬 Research Findings

### Why ResNet for Similarity (not DINOv2)?

| Aspect                | DINOv2          | ResNet50 + Contrastive          |
| --------------------- | --------------- | ------------------------------- |
| **Task**              | Classification  | Pairwise similarity             |
| **Training**          | Self-supervised | Triplet / contrastive loss      |
| **Output**            | Grade logits    | 128-dim L2-normalized embedding |
| **Similarity Metric** | Not ideal       | Cosine similarity ✅            |
| **Compute**           | ~100ms/img      | ~30ms/img ✅                    |

**Chosen:** ResNet50 fine-tuned with Siamese/triplet loss on produce image pairs (same-batch positive, different-batch negative).

### Similarity Thresholds (Business Logic)

| Score     | Interpretation                      | Action                         |
| --------- | ----------------------------------- | ------------------------------ |
| ≥ 0.90    | Same batch, no significant change   | Auto-settle                    |
| 0.70–0.89 | Same batch, moderate degradation    | Log + flag hauler              |
| 0.50–0.69 | Significant visual change           | HITL + liability investigation |
| < 0.50    | Possible substitution or total loss | Freeze payment + alert         |

---

## 🏗️ Implementation Spec

### 1. New Module `src/agents/digital_twin/similarity.py`

```python
"""
ResNet50 Image Similarity Engine for Digital Twin verification.
Computes cosine similarity between departure and arrival produce images.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD  = [0.229, 0.224, 0.225]
EMBED_DIM   = 128   # L2-normalized embedding output size


class ResNetSimilarityEngine:
    """
    Computes visual similarity between two produce images using
    a ResNet50 backbone fine-tuned with contrastive loss.
    """

    def __init__(self, model_dir: str = "models/vision/"):
        model_path = Path(model_dir) / "resnet50_produce_similarity.onnx"
        self.session = self._load_session(model_path)

    def _load_session(self, model_path: Path):
        if not model_path.exists():
            logger.warning("ResNet50 similarity model not found; using hash fallback")
            return None
        try:
            import onnxruntime as ort
            return ort.InferenceSession(str(model_path))
        except Exception as err:
            logger.warning(f"Failed loading ResNet50 model: {err}")
            return None

    @property
    def available(self) -> bool:
        return self.session is not None

    def _preprocess(self, image_bytes: bytes, size: int = 224) -> np.ndarray:
        """Preprocess bytes → (1, 3, 224, 224) float32 tensor."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((size, size), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array(RESNET_MEAN)[:, None, None]
        std  = np.array(RESNET_STD)[:, None, None]
        arr  = (arr.transpose(2, 0, 1) - mean) / std
        return arr[np.newaxis, :]

    def embed(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Extract 128-dim L2-normalized embedding for one image."""
        if self.session is None:
            return None
        tensor = self._preprocess(image_bytes)
        name   = self.session.get_inputs()[0].name
        embed  = self.session.run(None, {name: tensor})[0][0]   # (128,)
        norm   = np.linalg.norm(embed)
        return embed / norm if norm > 0 else embed

    def similarity(self, img_a: bytes, img_b: bytes) -> float:
        """
        Cosine similarity ∈ [0.0, 1.0] between two produce images.
        Falls back to perceptual hash comparison if model unavailable.
        """
        emb_a = self.embed(img_a)
        emb_b = self.embed(img_b)
        if emb_a is None or emb_b is None:
            return self._phash_similarity(img_a, img_b)
        return float(np.clip(np.dot(emb_a, emb_b), 0.0, 1.0))

    def compare_batches(
        self,
        departure_images: list[bytes],
        arrival_images: list[bytes],
    ) -> dict:
        """
        Compare entire departure vs arrival batch.
        Returns similarity_score (averaged), min_score, and substitution_flag.
        """
        if not departure_images or not arrival_images:
            return {"similarity_score": 0.5, "min_score": 0.5, "substitution_flag": False}

        scores = []
        for dep in departure_images[:3]:     # Sample up to 3 departure photos
            for arr in arrival_images[:3]:   # vs up to 3 arrival photos
                scores.append(self.similarity(dep, arr))

        avg_score = float(np.mean(scores))
        min_score = float(np.min(scores))
        return {
            "similarity_score": round(avg_score, 4),
            "min_score": round(min_score, 4),
            "substitution_flag": min_score < 0.50,
        }

    def _phash_similarity(self, img_a: bytes, img_b: bytes) -> float:
        """Fallback: perceptual hash similarity (no model required)."""
        try:
            import imagehash
            h1 = imagehash.phash(Image.open(io.BytesIO(img_a)))
            h2 = imagehash.phash(Image.open(io.BytesIO(img_b)))
            diff = h1 - h2         # Hamming distance [0, 64]
            return round(1.0 - diff / 64.0, 4)
        except Exception:
            return 0.70            # Neutral fallback
```

### 2. Integration into `DigitalTwinEngine.generate_diff_report`

```python
# In src/agents/digital_twin/engine.py — update generate_diff_report()
from src.agents.digital_twin.similarity import ResNetSimilarityEngine

class DigitalTwinEngine:
    def __init__(self, db=None):
        ...
        self.similarity_engine = ResNetSimilarityEngine()

    async def generate_diff_report(self, departure_twin, arrival_data):
        # Step 1: Decode arrival photos from base64
        arrival_bytes = [base64.b64decode(p) for p in arrival_data.photos if p]
        departure_bytes = [base64.b64decode(p) for p in departure_twin.agent_photos if p]

        # Step 2: ResNet batch comparison
        batch_result = self.similarity_engine.compare_batches(departure_bytes, arrival_bytes)
        similarity_score = batch_result["similarity_score"]
        substitution_flag = batch_result["substitution_flag"]

        # Step 3: Run arrival images through QA pipeline for actual grade
        if arrival_bytes and self.qa_pipeline:
            arrival_quality = await self.qa_pipeline.assess_quality(
                arrival_bytes[0], departure_twin.commodity
            )
            arrival_grade = arrival_quality.grade
        else:
            arrival_grade = _estimate_arrival_grade(departure_twin, arrival_data.photos, transit_hours)

        # Step 4: Determine liability using similarity + grade diff
        liability = determine_liability(
            similarity_score=similarity_score,
            departure_grade=departure_twin.quality_result.grade,
            arrival_grade=arrival_grade,
            substitution_flag=substitution_flag,
        )
        ...
```

### 3. Training Script Reference (`scripts/train_resnet_similarity.py`)

```python
"""
Fine-tune ResNet50 with triplet loss for produce image similarity.
Requires: torch, torchvision
Run: python scripts/train_resnet_similarity.py --data data/similarity/
"""
import torch, torch.nn as nn
from torchvision import models

class ProduceSimilarityNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        backbone = models.resnet50(weights="IMAGENET1K_V2")
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop classifier
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embed_dim),
            nn.functional.normalize  # L2 normalize
        )

    def forward(self, x):
        feat = self.features(x)
        return nn.functional.normalize(self.projection(feat), dim=1)
```

---

## ✅ Acceptance Criteria — All Met (2026-03-03)

| #   | Criterion                                                                                    | Weight | Result                                                                                            |
| --- | -------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| 1   | `ResNetSimilarityEngine.similarity(img_a, img_b)` returns float in [0.0, 1.0]                | 20%    | ✅ `TestSimilarity::test_similarity_bounded_zero_to_one`                                          |
| 2   | `compare_batches()` correctly averages pairwise scores across 3×3 image grid                 | 20%    | ✅ `TestCompareBatches::test_averages_pairwise_scores` — 9 calls, mean verified                   |
| 3   | `substitution_flag=True` when min pairwise score < 0.50                                      | 20%    | ✅ `TestCompareBatches::test_substitution_flag_true_when_min_below_threshold`                     |
| 4   | `generate_diff_report()` in `DigitalTwinEngine` uses ResNet score in liability determination | 20%    | ✅ `TestEngineIntegration::test_generate_diff_report_uses_resnet_when_available`                  |
| 5   | Graceful phash fallback when model file not present                                          | 10%    | ✅ `TestPhashFallback::test_phash_similarity_returns_valid_score`; smoke check `available: False` |
| 6   | Tests in `test_digital_twin_similarity.py` pass with mocked ONNX sessions                    | 10%    | ✅ **25/25 passed** — no GPU or real model required                                               |

---

## 📚 Dependencies

- `onnxruntime>=1.17.0` (existing)
- `Pillow>=10.0.0` (existing)
- `numpy>=1.24.0` (existing)
- `imagehash>=4.3.1` (new — phash fallback)
- Model file: `models/vision/resnet50_produce_similarity.onnx`
- **Training only:** `torch>=2.1`, `torchvision>=0.16`

---

## 📐 Related Files

- `src/agents/digital_twin/similarity.py` — new module (ResNet engine)
- `src/agents/digital_twin/engine.py` — update `generate_diff_report()`, inject `similarity_engine`
- `src/agents/digital_twin/liability.py` — update to accept `substitution_flag`
- `scripts/train_resnet_similarity.py` — training + ONNX export (new)
- `tests/unit/test_digital_twin_similarity.py` — new test file
- `models/vision/resnet50_produce_similarity.onnx` — output model artifact
