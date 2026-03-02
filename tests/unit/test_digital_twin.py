"""
Unit tests for the Digital Twin Engine.

Covers:
- Departure twin creation (in-memory, no DB)
- Diff report generation for various transit / grade scenarios
- Liability matrix decision tree
- Grade delta and similarity computations
- compare_arrival() and QualityAssessmentAgent.compare_twin() integration
"""

# * TEST MODULE — DIGITAL TWIN ENGINE
# NOTE: All tests run in-memory without DB or real image URLs.
# NOTE: Uses Arrange-Act-Assert (AAA) pattern throughout.

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.agents.digital_twin.diff_analysis import (
    compute_grade_delta,
    compute_new_defects,
    compute_rule_based_similarity,
    compute_similarity,
)
from src.agents.digital_twin.engine import (
    DigitalTwinEngine,
    _compute_transit_hours,
    _estimate_arrival_grade,
    _infer_arrival_defects,
    get_digital_twin_engine,
)
from src.agents.digital_twin.liability import (
    determine_liability,
    LONG_TRANSIT_HOURS,
    SHORT_TRANSIT_HOURS,
)
from src.agents.digital_twin.models import ArrivalData, DiffReport, DigitalTwin
from src.agents.quality_assessment.agent import QualityAssessmentAgent
from src.agents.quality_assessment.vision_models import QualityResult


# * ═══════════════════════════════════════════════════════════════
# * Fixtures
# * ═══════════════════════════════════════════════════════════════

@pytest.fixture
def engine() -> DigitalTwinEngine:
    """In-memory DigitalTwinEngine with no DB dependency."""
    return get_digital_twin_engine(db=None)


@pytest.fixture
def quality_result_grade_a() -> QualityResult:
    """QualityResult fixture for a Grade A assessment."""
    return QualityResult(
        grade="A",
        confidence=0.82,
        defects=["bruise"],
        defect_count=1,
        hitl_required=False,
        annotations=[],
        shelf_life_days=5,
        assessment_mode="rule_based",
    )


@pytest.fixture
def quality_result_grade_a_plus() -> QualityResult:
    """QualityResult fixture for a premium Grade A+ assessment."""
    return QualityResult(
        grade="A+",
        confidence=0.91,
        defects=[],
        defect_count=0,
        hitl_required=False,
        annotations=[],
        shelf_life_days=7,
        assessment_mode="rule_based",
    )


@pytest.fixture
def sample_twin(engine: DigitalTwinEngine) -> DigitalTwin:
    """Pre-built DigitalTwin cached in the engine for reuse across tests."""
    twin = DigitalTwin(
        twin_id="dt-testfixture001",
        listing_id="lst-test001",
        farmer_photos=["s3://bucket/farm1.jpg"],
        agent_photos=["s3://bucket/agent1.jpg"],
        grade="A",
        confidence=0.82,
        defect_types=["bruise"],
        defect_count=1,
        shelf_life_days=5,
        gps_lat=12.9716,
        gps_lng=77.5946,
        ai_annotations={"bboxes": []},
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )
    engine._twin_cache[twin.twin_id] = twin
    return twin


# * ═══════════════════════════════════════════════════════════════
# * Grade Delta Tests
# * ═══════════════════════════════════════════════════════════════

class TestGradeDelta:

    def test_no_change_returns_zero(self):
        """AAA: same grade at departure and arrival → delta = 0.0"""
        # Arrange / Act
        delta = compute_grade_delta("A", "A")
        # Assert
        assert delta == 0.0

    def test_one_grade_drop(self):
        """A → B should produce a negative delta."""
        delta = compute_grade_delta("A", "B")
        assert delta < 0.0
        assert delta > -1.0

    def test_maximum_drop(self):
        """A+ → C should be close to -1.0."""
        delta = compute_grade_delta("A+", "C")
        assert delta <= -0.9

    def test_improvement_clamped_to_zero(self):
        """B → A should be clamped to 0.0 (no negative delta for improvement)."""
        delta = compute_grade_delta("B", "A")
        assert delta == 0.0

    def test_unknown_grade_does_not_raise(self):
        """Unknown grade strings should not raise exceptions."""
        delta = compute_grade_delta("X", "Y")
        assert isinstance(delta, float)


# * ═══════════════════════════════════════════════════════════════
# * New Defect Detection Tests
# * ═══════════════════════════════════════════════════════════════

class TestNewDefects:

    def test_no_new_defects_when_identical(self):
        """Same defects at departure and arrival → empty new defects."""
        departure = ["bruise", "colour_off"]
        arrival = ["bruise", "colour_off"]
        new = compute_new_defects(departure, arrival)
        assert new == []

    def test_detects_new_arrival_defects(self):
        """rot_spot at arrival but not departure → detected as new."""
        departure = ["bruise"]
        arrival = ["bruise", "rot_spot"]
        new = compute_new_defects(departure, arrival)
        assert "rot_spot" in new
        assert "bruise" not in new

    def test_empty_departure_all_arrival_are_new(self):
        """Pristine departure, all defects at arrival are new."""
        departure: list[str] = []
        arrival = ["bruise", "overripe"]
        new = compute_new_defects(departure, arrival)
        assert set(new) == {"bruise", "overripe"}


# * ═══════════════════════════════════════════════════════════════
# * Rule-Based Similarity Tests
# * ═══════════════════════════════════════════════════════════════

class TestRuleBasedSimilarity:

    def test_identical_grade_no_new_defects_high_similarity(self):
        """Same grade and defects should yield similarity close to 1.0."""
        score = compute_rule_based_similarity("A", "A", ["bruise"], ["bruise"])
        assert score >= 0.90

    def test_grade_drop_lowers_similarity(self):
        """One grade drop should meaningfully reduce similarity."""
        same_grade = compute_rule_based_similarity("A", "A", [], [])
        dropped = compute_rule_based_similarity("A", "B", [], [])
        assert dropped < same_grade

    def test_score_bounded(self):
        """Similarity must always be in [0.0, 1.0]."""
        score = compute_rule_based_similarity("A+", "C", ["worm_hole", "rot_spot"], ["bruise", "overripe", "rot_spot"])
        assert 0.0 <= score <= 1.0


# * ═══════════════════════════════════════════════════════════════
# * Compute Similarity Dispatcher Tests
# * ═══════════════════════════════════════════════════════════════

class TestComputeSimilarity:

    def test_falls_back_to_rule_based_without_photos(self):
        """Empty photo lists → rule_based method used."""
        score, method = compute_similarity(
            departure_photos=[],
            arrival_photos=[],
            grade_departure="A",
            grade_arrival="B",
            departure_defects=[],
            arrival_defects=[],
        )
        assert method == "rule_based"
        assert 0.0 <= score <= 1.0

    def test_falls_back_to_rule_based_for_non_http_urls(self):
        """File paths that aren't HTTP URLs → cannot fetch → rule_based."""
        score, method = compute_similarity(
            departure_photos=["s3://my-bucket/photo.jpg"],
            arrival_photos=["s3://my-bucket/arrival.jpg"],
            grade_departure="A",
            grade_arrival="A",
            departure_defects=[],
            arrival_defects=[],
        )
        assert method == "rule_based"
        assert 0.0 <= score <= 1.0

    def test_returns_float_and_method_string(self):
        """Return types are always (float, str) regardless of input."""
        score, method = compute_similarity([], [], "A", "C", [], ["bruise"])
        assert isinstance(score, float)
        assert isinstance(method, str)


# * ═══════════════════════════════════════════════════════════════
# * Transit Hours Helper Tests
# * ═══════════════════════════════════════════════════════════════

class TestTransitHours:

    def test_4_hour_transit(self):
        """Standard 4-hour delivery window computed correctly."""
        departed = datetime(2026, 3, 1, 6, 0)
        arrived = datetime(2026, 3, 1, 10, 0)
        assert _compute_transit_hours(departed, arrived) == 4.0

    def test_negative_delta_clamped_to_zero(self):
        """Arrival before departure (clock drift) → clamped to 0."""
        departed = datetime(2026, 3, 1, 10, 0)
        arrived = datetime(2026, 3, 1, 8, 0)
        assert _compute_transit_hours(departed, arrived) == 0.0

    def test_very_long_transit_clamped(self):
        """Transit > 72h clamped to 72.0."""
        departed = datetime(2026, 3, 1, 0, 0)
        arrived = datetime(2026, 4, 1, 0, 0)
        assert _compute_transit_hours(departed, arrived) == 72.0


# * ═══════════════════════════════════════════════════════════════
# * Arrival Grade Estimation Tests
# * ═══════════════════════════════════════════════════════════════

class TestArrivalGradeEstimation:

    def test_no_photos_returns_c(self, sample_twin: DigitalTwin):
        """No arrival photos → worst-case grade C (for liability protection)."""
        # Arrange: twin with 5-day shelf life
        grade = _estimate_arrival_grade(sample_twin, [], transit_hours=2.0)
        assert grade == "C"

    def test_short_transit_preserves_grade(self, sample_twin: DigitalTwin):
        """Very short transit (<10% shelf life) → no grade drop expected."""
        # 5-day shelf life = 120h; 10% = 12h. Transit of 2h = 1.67% consumed.
        grade = _estimate_arrival_grade(sample_twin, ["s3://bucket/arrival.jpg"], transit_hours=2.0)
        assert grade == sample_twin.grade  # Grade A preserved

    def test_long_transit_degrades_grade(self, sample_twin: DigitalTwin):
        """Transit consuming >25% shelf life → at least one grade drop."""
        # 5-day shelf life = 120h. 40h = 33% consumed → 1 grade drop expected.
        grade = _estimate_arrival_grade(sample_twin, ["s3://bucket/arrival.jpg"], transit_hours=40.0)
        # Grade A should drop to B
        assert grade in ("B", "C")


# * ═══════════════════════════════════════════════════════════════
# * Infer Arrival Defects Tests
# * ═══════════════════════════════════════════════════════════════

class TestInferArrivalDefects:

    def test_grade_b_adds_bruise(self):
        """Grade B arrival should have 'bruise' added if not already present."""
        defects = _infer_arrival_defects("B", [])
        assert "bruise" in defects

    def test_grade_c_adds_bruise_and_overripe(self):
        """Grade C arrival should add both 'bruise' and 'overripe'."""
        defects = _infer_arrival_defects("C", [])
        assert "bruise" in defects
        assert "overripe" in defects

    def test_does_not_duplicate_existing_defects(self):
        """Should not add defects already present in departure."""
        defects = _infer_arrival_defects("B", ["bruise", "rot_spot"])
        assert defects.count("bruise") == 1

    def test_grade_a_no_extra_defects(self):
        """Grade A arrival should not add transit-induced defects."""
        defects = _infer_arrival_defects("A", ["colour_off"])
        assert "bruise" not in defects
        assert "overripe" not in defects


# * ═══════════════════════════════════════════════════════════════
# * Liability Matrix Tests
# * ═══════════════════════════════════════════════════════════════

class TestLiabilityMatrix:

    def test_no_photos_claim_rejected(self):
        """Rule 1: buyer submits no arrival photos → claim rejected."""
        result = determine_liability(
            grade_departure="A", grade_arrival="B",
            quality_delta=-0.33, transit_hours=4.0,
            new_defects=["bruise"], has_arrival_photos=False,
        )
        assert result.liable_party == "none"
        assert result.claim_percent == 0.0
        assert "photos" in result.reasoning.lower()

    def test_no_degradation_no_liability(self):
        """Rule 2: same grade, no new defects → no liability."""
        result = determine_liability(
            grade_departure="A", grade_arrival="A",
            quality_delta=0.0, transit_hours=4.0,
            new_defects=[], has_arrival_photos=True,
        )
        assert result.liable_party == "none"
        assert result.claim_percent == 0.0

    def test_long_transit_hauler_liable(self):
        """Rule 4: grade drop + transit > 6h → hauler responsible."""
        result = determine_liability(
            grade_departure="A", grade_arrival="B",
            quality_delta=-0.33, transit_hours=8.0,
            new_defects=["bruise"], has_arrival_photos=True,
        )
        assert result.liable_party == "hauler"
        assert result.claim_percent > 0.0
        assert "hauler" in result.reasoning.lower()

    def test_short_transit_farmer_liable(self):
        """Rule 5: grade drop + transit < 2h → farmer responsible (pre-existing)."""
        result = determine_liability(
            grade_departure="A", grade_arrival="B",
            quality_delta=-0.33, transit_hours=1.0,
            new_defects=["bruise"], has_arrival_photos=True,
        )
        assert result.liable_party == "farmer"
        assert "farmer" in result.reasoning.lower()

    def test_mid_transit_shared_liability(self):
        """Rule 6: transit 2–6h → shared liability."""
        result = determine_liability(
            grade_departure="A", grade_arrival="B",
            quality_delta=-0.33, transit_hours=4.0,
            new_defects=["bruise"], has_arrival_photos=True,
        )
        assert result.liable_party == "shared"

    def test_quantity_mismatch_above_threshold(self):
        """Rule 3: quantity mismatch > 5% → shared liability."""
        result = determine_liability(
            grade_departure="A", grade_arrival="A",
            quality_delta=0.0, transit_hours=3.0,
            new_defects=[], has_arrival_photos=True,
            quantity_mismatch_percent=8.0,
        )
        assert result.liable_party == "shared"
        assert result.claim_percent > 0.0

    def test_claim_percent_bounded(self):
        """Claim % must always be in [0, 100]."""
        for transit_hours in [0.5, 3.0, 8.0]:
            result = determine_liability(
                grade_departure="A+", grade_arrival="C",
                quality_delta=-1.0, transit_hours=transit_hours,
                new_defects=["bruise", "overripe", "rot_spot", "worm_hole"],
                has_arrival_photos=True,
            )
            assert 0.0 <= result.claim_percent <= 100.0


# * ═══════════════════════════════════════════════════════════════
# * DigitalTwinEngine End-to-End Tests
# * ═══════════════════════════════════════════════════════════════

class TestDigitalTwinEngine:

    @pytest.mark.asyncio
    async def test_create_departure_twin_returns_twin(
        self, engine: DigitalTwinEngine, quality_result_grade_a: QualityResult
    ):
        """Departure twin creation should return a DigitalTwin with correct fields."""
        # Arrange
        listing_id = "lst-engine-test-001"
        farmer_photos = ["s3://bucket/farm1.jpg", "s3://bucket/farm2.jpg"]
        agent_photos = ["s3://bucket/agent1.jpg"]
        gps = (12.9716, 77.5946)

        # Act
        twin = await engine.create_departure_twin(
            listing_id=listing_id,
            farmer_photos=farmer_photos,
            agent_photos=agent_photos,
            quality_result=quality_result_grade_a,
            gps=gps,
        )

        # Assert
        assert twin.twin_id.startswith("dt-")
        assert twin.listing_id == listing_id
        assert twin.grade == "A"
        assert twin.confidence == 0.82
        assert twin.defect_types == ["bruise"]
        assert twin.shelf_life_days == 5
        assert twin.gps_lat == 12.9716
        assert twin.gps_lng == 77.5946
        assert twin.farmer_photos == farmer_photos
        assert twin.agent_photos == agent_photos

    @pytest.mark.asyncio
    async def test_twin_cached_after_creation(
        self, engine: DigitalTwinEngine, quality_result_grade_a: QualityResult
    ):
        """Twin should be retrievable from cache after creation."""
        twin = await engine.create_departure_twin(
            listing_id="lst-cache-test",
            farmer_photos=[],
            agent_photos=[],
            quality_result=quality_result_grade_a,
            gps=(0.0, 0.0),
        )
        assert twin.twin_id in engine._twin_cache

    @pytest.mark.asyncio
    async def test_compare_arrival_returns_diff_report(
        self, engine: DigitalTwinEngine, sample_twin: DigitalTwin
    ):
        """compare_arrival() should return a DiffReport for a valid twin."""
        # Act
        diff = await engine.compare_arrival(
            twin_id=sample_twin.twin_id,
            arrival_photos=["s3://bucket/arrival.jpg"],
            arrival_gps=(12.9700, 77.5900),
        )

        # Assert
        assert isinstance(diff, DiffReport)
        assert diff.grade_departure == "A"
        assert diff.grade_arrival in ("A+", "A", "B", "C")
        assert 0.0 <= diff.similarity_score <= 1.0
        assert 0.0 <= diff.confidence <= 1.0
        assert diff.liability in ("farmer", "hauler", "buyer", "shared", "none")
        assert 0.0 <= diff.claim_percent <= 100.0
        assert isinstance(diff.explanation, str)
        assert len(diff.explanation) > 10

    @pytest.mark.asyncio
    async def test_compare_arrival_raises_for_unknown_twin(
        self, engine: DigitalTwinEngine
    ):
        """compare_arrival() should raise ValueError for an unknown twin_id."""
        with pytest.raises(ValueError, match="not found"):
            await engine.compare_arrival(
                twin_id="dt-nonexistent999",
                arrival_photos=["s3://bucket/img.jpg"],
            )

    @pytest.mark.asyncio
    async def test_diff_report_no_photos_claim_zero(
        self, engine: DigitalTwinEngine, sample_twin: DigitalTwin
    ):
        """No arrival photos → liability 'none' and claim_percent = 0."""
        diff = await engine.compare_arrival(
            twin_id=sample_twin.twin_id,
            arrival_photos=[],   # ! No photos submitted
        )
        assert diff.liability == "none"
        assert diff.claim_percent == 0.0

    @pytest.mark.asyncio
    async def test_generate_diff_report_short_transit_no_drop(
        self, engine: DigitalTwinEngine, quality_result_grade_a_plus: QualityResult
    ):
        """Premium A+ produce on 1h transit should show minimal degradation."""
        # Arrange
        twin = await engine.create_departure_twin(
            listing_id="lst-premium-001",
            farmer_photos=["s3://bucket/farm.jpg"],
            agent_photos=[],
            quality_result=quality_result_grade_a_plus,
            gps=(12.0, 77.0),
        )
        # Simulate 1h transit (very fresh delivery)
        arrival = ArrivalData(
            arrival_photos=["s3://bucket/arrival.jpg"],
            gps_lat=12.0,
            gps_lng=77.0,
            arrived_at=twin.created_at + timedelta(hours=1),
        )

        # Act
        diff = await engine.generate_diff_report(twin, arrival)

        # Assert: 1h transit on 7-day shelf-life produce → no grade drop
        assert diff.grade_departure == "A+"
        assert diff.grade_arrival == "A+"
        assert diff.quality_delta == 0.0
        assert diff.transit_hours == pytest.approx(1.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_generate_diff_report_long_transit_hauler_blamed(
        self, engine: DigitalTwinEngine, quality_result_grade_a: QualityResult
    ):
        """Long transit (>6h) on produce with grade drop → hauler blamed."""
        # Arrange
        twin = await engine.create_departure_twin(
            listing_id="lst-long-transit-001",
            farmer_photos=["s3://bucket/farm.jpg"],
            agent_photos=[],
            quality_result=quality_result_grade_a,
            gps=(12.0, 77.0),
        )
        # Simulate 8-hour transit on 5-day shelf-life (16.7% consumed → 1 grade drop)
        arrival = ArrivalData(
            arrival_photos=["s3://bucket/arrival.jpg"],
            gps_lat=12.0,
            gps_lng=77.0,
            arrived_at=twin.created_at + timedelta(hours=35),  # 35h > 25% of 120h shelf life
        )

        # Act
        diff = await engine.generate_diff_report(twin, arrival)

        # Assert: grade should drop and hauler blamed (>6h transit)
        assert diff.transit_hours > LONG_TRANSIT_HOURS
        assert diff.grade_arrival in ("B", "C")
        assert diff.liability == "hauler"

    @pytest.mark.asyncio
    async def test_diff_report_to_dict_is_json_serializable(
        self, engine: DigitalTwinEngine, sample_twin: DigitalTwin
    ):
        """DiffReport.to_dict() must produce a JSON-serializable dict."""
        import json as stdlib_json

        diff = await engine.compare_arrival(
            twin_id=sample_twin.twin_id,
            arrival_photos=["s3://bucket/img.jpg"],
        )
        payload = diff.to_dict()
        json_str = stdlib_json.dumps(payload)
        parsed = stdlib_json.loads(json_str)
        assert parsed["liability"] == diff.liability
        assert parsed["claim_percent"] == diff.claim_percent


# * Extended tests moved to test_digital_twin_integration.py (Modular_Coding 500-line rule)
