"""
Unit tests — ResNet50 Digital Twin Similarity Engine
=====================================================
Covers all 6 acceptance criteria from task33.md:
    1. similarity() returns float in [0.0, 1.0]
    2. compare_batches() correctly averages pairwise scores (3×3 grid)
    3. substitution_flag=True when min pairwise score < 0.50
    4. generate_diff_report() uses ResNet score in liability determination
    5. Graceful phash fallback when model file not present
    6. All tests pass with mocked ONNX sessions (no real model required)
"""

# * TEST MODULE — RESNET SIMILARITY ENGINE
# NOTE: No real ONNX model or GPU needed — all inference is mocked.
# NOTE: Uses Arrange-Act-Assert (AAA) pattern throughout.

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.agents.digital_twin.similarity import (
    ResNetSimilarityEngine,
    SUBSTITUTION_THRESHOLD,
    _build_batch_result,
    _fetch_url_list,
)


# * ═══════════════════════════════════════════════════════════════
# * Fixture helpers
# * ═══════════════════════════════════════════════════════════════

def _make_image_bytes(color: tuple[int, int, int] = (128, 200, 64)) -> bytes:
    """Create a minimal valid JPEG image as bytes for testing."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="JPEG")
    return buf.getvalue()


def _make_mock_session(embed: np.ndarray) -> MagicMock:
    """Return a mock ONNX InferenceSession that emits a fixed embedding."""
    mock_input = MagicMock()
    mock_input.name = "image"
    session = MagicMock()
    session.get_inputs.return_value = [mock_input]
    # * ONNX session.run returns list of arrays; first array shape (1, embed_dim)
    session.run.return_value = [embed[np.newaxis, :]]
    return session


# * ═══════════════════════════════════════════════════════════════
# * TestResNetSimilarityEngine — construction + availability
# * ═══════════════════════════════════════════════════════════════

class TestResNetSimilarityEngineConstruction:

    def test_available_false_when_model_missing(self):
        """AC 5 + 6: No ONNX file → available=False, no crash."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent_path_xyzzy/")
        assert engine.available is False
        assert engine.session is None

    def test_available_true_when_session_injected(self):
        """Engine is available when session is manually set (mocking load)."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent_path_xyzzy/")
        engine.session = _make_mock_session(np.ones(128))
        assert engine.available is True


# * ═══════════════════════════════════════════════════════════════
# * TestPreprocess — tensor shape and normalisation
# * ═══════════════════════════════════════════════════════════════

class TestPreprocess:

    def test_output_shape(self):
        """Preprocessed tensor should be (1, 3, 224, 224) for ResNet input."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        img_bytes = _make_image_bytes()
        tensor = engine._preprocess(img_bytes)
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_dtype_float32(self):
        """Preprocessed tensor must be float32 for ONNX/numpy compatibility."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        tensor = engine._preprocess(_make_image_bytes())
        assert tensor.dtype == np.float32


# * ═══════════════════════════════════════════════════════════════
# * TestEmbed — embedding extraction with mocked ONNX session
# * ═══════════════════════════════════════════════════════════════

class TestEmbed:

    def test_embed_returns_none_when_session_is_none(self):
        """AC 5: embed() returns None gracefully when model not loaded."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        assert engine.embed(_make_image_bytes()) is None

    def test_embed_returns_l2_normalized_vector(self):
        """AC 6: embed() output should be L2-normalised (norm ≈ 1.0)."""
        raw = np.random.rand(128).astype(np.float32)
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        engine.session = _make_mock_session(raw)

        emb = engine.embed(_make_image_bytes())
        assert emb is not None
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    def test_embed_shape(self):
        """Embedding must be a 1-D array of length EMBED_DIM (128)."""
        raw = np.random.rand(128).astype(np.float32)
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        engine.session = _make_mock_session(raw)

        emb = engine.embed(_make_image_bytes())
        assert emb is not None
        assert emb.shape == (128,)


# * ═══════════════════════════════════════════════════════════════
# * TestSimilarity — cosine similarity computation
# * ═══════════════════════════════════════════════════════════════

class TestSimilarity:

    def test_identical_embeddings_score_one(self):
        """AC 1: cosine similarity of a vector with itself = 1.0."""
        vec = np.random.rand(128).astype(np.float32)
        norm = vec / np.linalg.norm(vec)
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        engine.session = _make_mock_session(norm)

        score = engine.similarity(_make_image_bytes(), _make_image_bytes())
        assert abs(score - 1.0) < 1e-4

    def test_orthogonal_embeddings_score_zero(self):
        """AC 1: near-orthogonal vectors → similarity ≈ 0."""
        # Build two orthogonal unit vectors in 128-dim space
        vec_a = np.zeros(128, dtype=np.float32); vec_a[0] = 1.0
        vec_b = np.zeros(128, dtype=np.float32); vec_b[64] = 1.0

        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        # ? We need to patch embed() to return the two different vectors
        with patch.object(engine, "embed", side_effect=[vec_a, vec_b]):
            score = engine.similarity(_make_image_bytes(), _make_image_bytes())

        assert score < 0.01   # Orthogonal → cosine = 0

    def test_similarity_bounded_zero_to_one(self):
        """AC 1: output must always be in [0.0, 1.0] (np.clip guard)."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        # Simulate anti-correlated embeddings (dot product < 0) — should be clipped to 0
        vec_a =  np.ones(128, dtype=np.float32) / np.sqrt(128)
        vec_b = -np.ones(128, dtype=np.float32) / np.sqrt(128)
        with patch.object(engine, "embed", side_effect=[vec_a, vec_b]):
            score = engine.similarity(_make_image_bytes(), _make_image_bytes())
        assert 0.0 <= score <= 1.0

    def test_similarity_falls_back_to_phash_when_no_session(self):
        """AC 5: When model absent, similarity() calls _phash_similarity."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        assert engine.session is None

        with patch.object(engine, "_phash_similarity", return_value=0.85) as mock_phash:
            score = engine.similarity(_make_image_bytes(), _make_image_bytes())

        mock_phash.assert_called_once()
        assert score == 0.85


# * ═══════════════════════════════════════════════════════════════
# * TestCompareBatches — batch comparison (3×3 grid)
# * ═══════════════════════════════════════════════════════════════

class TestCompareBatches:

    def _engine_with_fixed_score(self, fixed_score: float) -> ResNetSimilarityEngine:
        """Engine that returns a constant similarity score for all pairs."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        engine.session = MagicMock()   # Makes available=True
        engine.similarity = MagicMock(return_value=fixed_score)  # type: ignore[method-assign]
        return engine

    def test_empty_inputs_return_neutral(self):
        """AC 2: Empty departure or arrival images → neutral 0.5 score, no flag."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        result = engine.compare_batches([], [])
        assert result["similarity_score"] == 0.5
        assert result["substitution_flag"] is False

    def test_averages_pairwise_scores(self):
        """AC 2: compare_batches() returns mean across all pairwise combinations."""
        engine = self._engine_with_fixed_score(0.80)
        dep = [_make_image_bytes()] * 3
        arr = [_make_image_bytes()] * 3
        result = engine.compare_batches(dep, arr)

        # 3×3 = 9 calls, all returning 0.80 → average = 0.80
        assert engine.similarity.call_count == 9
        assert abs(result["similarity_score"] - 0.80) < 1e-4

    def test_samples_max_three_images_per_side(self):
        """AC 2: Only the first 3 images of each list are sampled."""
        engine = self._engine_with_fixed_score(0.75)
        dep = [_make_image_bytes()] * 5   # 5 images, only 3 should be used
        arr = [_make_image_bytes()] * 4   # 4 images, only 3 should be used
        engine.compare_batches(dep, arr)
        assert engine.similarity.call_count == 9   # 3×3 capped

    def test_substitution_flag_true_when_min_below_threshold(self):
        """AC 3: substitution_flag=True when any pairwise score < 0.50."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        engine.session = MagicMock()
        # Mixed scores: some normal, one very low (substitution suspect)
        scores_iter = iter([0.90, 0.85, 0.40, 0.88])   # min=0.40 → flag
        engine.similarity = MagicMock(side_effect=scores_iter)  # type: ignore[method-assign]

        dep = [_make_image_bytes()] * 2
        arr = [_make_image_bytes()] * 2
        result = engine.compare_batches(dep, arr)

        assert result["substitution_flag"] is True
        assert result["min_score"] == pytest.approx(0.40, abs=1e-4)

    def test_substitution_flag_false_when_all_scores_above_threshold(self):
        """AC 3: substitution_flag=False when all pairwise scores ≥ 0.50."""
        engine = self._engine_with_fixed_score(0.75)
        dep = [_make_image_bytes()] * 2
        arr = [_make_image_bytes()] * 2
        result = engine.compare_batches(dep, arr)
        assert result["substitution_flag"] is False


# * ═══════════════════════════════════════════════════════════════
# * TestPhashFallback — perceptual hash path
# * ═══════════════════════════════════════════════════════════════

class TestPhashFallback:

    def test_phash_similarity_returns_valid_score(self):
        """AC 5: _phash_similarity returns a float in [0.0, 1.0] when imagehash is installed."""
        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        img_bytes = _make_image_bytes()

        # * Use the real imagehash package (installed via pyproject.toml imagehash>=4.3.1)
        # * If it's not installed the test gracefully verifies the neutral fallback value
        try:
            import imagehash  # noqa: F401
            score = engine._phash_similarity(img_bytes, img_bytes)
            # Identical image → distance = 0 → score should be exactly 1.0
            assert abs(score - 1.0) < 1e-4
        except ImportError:
            # imagehash not installed in this environment — neutral fallback
            score = engine._phash_similarity(img_bytes, img_bytes)
            assert score == 0.70

    def test_phash_returns_neutral_on_exception(self):
        """AC 5: When imagehash raises any exception, neutral 0.70 is returned."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "imagehash":
                raise ImportError("imagehash not available")
            return real_import(name, *args, **kwargs)

        engine = ResNetSimilarityEngine(model_dir="models/nonexistent/")
        img_bytes = _make_image_bytes()

        with patch("builtins.__import__", side_effect=mock_import):
            score = engine._phash_similarity(img_bytes, img_bytes)

        assert score == 0.70


# * ═══════════════════════════════════════════════════════════════
# * TestBuildBatchResult — pure helper
# * ═══════════════════════════════════════════════════════════════

class TestBuildBatchResult:

    def test_mean_and_min_computed_correctly(self):
        scores = [0.90, 0.80, 0.70]
        result = _build_batch_result(scores)
        assert abs(result["similarity_score"] - 0.80) < 1e-4
        assert abs(result["min_score"] - 0.70) < 1e-4

    def test_substitution_flag_boundary(self):
        """Flag exactly at threshold boundary: 0.50 should be False (strictly less than)."""
        result_at = _build_batch_result([SUBSTITUTION_THRESHOLD])
        result_below = _build_batch_result([SUBSTITUTION_THRESHOLD - 0.001])
        assert result_at["substitution_flag"] is False
        assert result_below["substitution_flag"] is True


# * ═══════════════════════════════════════════════════════════════
# * TestFetchUrlList — URL loading helper
# * ═══════════════════════════════════════════════════════════════

class TestFetchUrlList:

    def test_non_http_urls_skipped(self):
        """S3 URLs and file paths are not HTTP → returned list is empty."""
        result = _fetch_url_list(["s3://my-bucket/photo.jpg", "/tmp/local.jpg"])
        assert result == []

    def test_empty_input_returns_empty(self):
        assert _fetch_url_list([]) == []


# * ═══════════════════════════════════════════════════════════════
# * TestEngineIntegration — generate_diff_report wires ResNet score
# * ═══════════════════════════════════════════════════════════════

class TestEngineIntegration:

    @pytest.mark.asyncio
    async def test_generate_diff_report_uses_resnet_when_available(self):
        """AC 4: DiffReport.similarity_score reflects ResNet batch result."""
        from src.agents.digital_twin.engine import DigitalTwinEngine
        from src.agents.digital_twin.models import ArrivalData, DigitalTwin

        engine = DigitalTwinEngine(db=None)
        # * Inject a mock similarity engine that returns a known high score
        engine.similarity_engine = MagicMock()
        engine.similarity_engine.available = True
        engine.similarity_engine.compare_url_batches.return_value = {
            "similarity_score": 0.92,
            "min_score":        0.88,
            "substitution_flag": False,
        }

        twin = DigitalTwin(
            twin_id="dt-integ-test-001",
            listing_id="lst-001",
            farmer_photos=["s3://bucket/farm.jpg"],
            agent_photos=["s3://bucket/agent.jpg"],
            grade="A",
            confidence=0.88,
            defect_types=[],
            defect_count=0,
            shelf_life_days=5,
            gps_lat=0.0,
            gps_lng=0.0,
            ai_annotations={},
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
        engine._twin_cache[twin.twin_id] = twin

        arrival = ArrivalData(
            arrival_photos=["s3://bucket/arrival.jpg"],
            gps_lat=0.0,
            gps_lng=0.0,
            arrived_at=twin.created_at + timedelta(hours=2),
        )

        diff = await engine.generate_diff_report(twin, arrival)

        # AC 4: ResNet score used in the report
        assert diff.similarity_score == pytest.approx(0.92, abs=1e-4)
        assert diff.analysis_method == "resnet50"

    @pytest.mark.asyncio
    async def test_generate_diff_report_substitution_flag_escalates_liability(self):
        """AC 4: substitution_flag=True → liability='hauler', claim_percent=100.0."""
        from src.agents.digital_twin.engine import DigitalTwinEngine
        from src.agents.digital_twin.models import ArrivalData, DigitalTwin

        engine = DigitalTwinEngine(db=None)
        engine.similarity_engine = MagicMock()
        engine.similarity_engine.available = True
        engine.similarity_engine.compare_url_batches.return_value = {
            "similarity_score": 0.30,
            "min_score":        0.25,
            "substitution_flag": True,   # ! Suspicious — triggers Rule 0
        }

        twin = DigitalTwin(
            twin_id="dt-sub-test-002",
            listing_id="lst-002",
            farmer_photos=["s3://bucket/farm.jpg"],
            agent_photos=[],
            grade="A+",
            confidence=0.95,
            defect_types=[],
            defect_count=0,
            shelf_life_days=7,
            gps_lat=0.0,
            gps_lng=0.0,
            ai_annotations={},
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
        engine._twin_cache[twin.twin_id] = twin

        arrival = ArrivalData(
            arrival_photos=["s3://bucket/arrival.jpg"],
            gps_lat=0.0,
            gps_lng=0.0,
            arrived_at=twin.created_at + timedelta(hours=4),
        )

        diff = await engine.generate_diff_report(twin, arrival)

        assert diff.liability == "hauler"
        assert diff.claim_percent == pytest.approx(100.0)
        assert "substitution" in diff.explanation.lower()

    @pytest.mark.asyncio
    async def test_generate_diff_report_fallback_without_resnet(self):
        """When ResNet not available, DiffReport still succeeds via diff_analysis fallback."""
        from src.agents.digital_twin.engine import DigitalTwinEngine
        from src.agents.digital_twin.models import ArrivalData, DigitalTwin

        engine = DigitalTwinEngine(db=None)
        # * Ensure similarity_engine is NOT available
        engine.similarity_engine = MagicMock()
        engine.similarity_engine.available = False

        twin = DigitalTwin(
            twin_id="dt-fallback-003",
            listing_id="lst-003",
            farmer_photos=[],
            agent_photos=[],
            grade="B",
            confidence=0.70,
            defect_types=["bruise"],
            defect_count=1,
            shelf_life_days=3,
            gps_lat=0.0,
            gps_lng=0.0,
            ai_annotations={},
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
        engine._twin_cache[twin.twin_id] = twin

        arrival = ArrivalData(
            arrival_photos=["s3://bucket/arrival.jpg"],
            gps_lat=0.0,
            gps_lng=0.0,
            arrived_at=twin.created_at + timedelta(hours=3),
        )

        diff = await engine.generate_diff_report(twin, arrival)

        assert diff.analysis_method != "resnet50"   # Fell back to rule_based / ssim
        assert 0.0 <= diff.similarity_score <= 1.0
        assert isinstance(diff.explanation, str)
