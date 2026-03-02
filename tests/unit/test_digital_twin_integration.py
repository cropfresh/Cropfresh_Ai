"""
Unit tests — Digital Twin QualityAssessmentAgent integration and parametrized cases.

* TEST MODULE: QA agent delegation, arrival grade parametrized
NOTE: Extracted from test_digital_twin.py to keep files under 500 lines (Modular_Coding).
"""

from datetime import datetime

import pytest

from src.agents.digital_twin.engine import (
    _estimate_arrival_grade,
    get_digital_twin_engine,
)
from src.agents.digital_twin.models import DigitalTwin
from src.agents.quality_assessment.agent import QualityAssessmentAgent
from src.agents.quality_assessment.vision_models import QualityResult


# * ═══════════════════════════════════════════════════════════════
# * QualityAssessmentAgent Twin Integration
# * ═══════════════════════════════════════════════════════════════

class TestQualityAgentTwinIntegration:
    """Agent.create_departure_twin and compare_twin delegation."""

    @pytest.fixture
    def qa_agent(self) -> QualityAssessmentAgent:
        return QualityAssessmentAgent(llm=None)

    @pytest.mark.asyncio
    async def test_create_departure_twin_via_agent(
        self, qa_agent: QualityAssessmentAgent
    ):
        """Agent.create_departure_twin() should delegate to twin_engine."""
        qr = QualityResult(
            grade="B", confidence=0.68, defects=["bruise", "colour_off"],
            defect_count=2, hitl_required=True, annotations=[],
            shelf_life_days=3, assessment_mode="rule_based",
        )
        twin = await qa_agent.create_departure_twin(
            listing_id="lst-qa-agent-001",
            farmer_photos=["s3://bucket/farm.jpg"],
            agent_photos=[],
            quality_result=qr,
        )
        assert twin.grade == "B"
        assert twin.defect_count == 2
        assert twin.twin_id.startswith("dt-")

    @pytest.mark.asyncio
    async def test_compare_twin_via_agent(self, qa_agent: QualityAssessmentAgent):
        """Agent.compare_twin() should return a DiffReport."""
        from src.agents.digital_twin.models import DiffReport

        qr = QualityResult(
            grade="A", confidence=0.80, defects=[],
            defect_count=0, hitl_required=False, annotations=[],
            shelf_life_days=5, assessment_mode="rule_based",
        )
        twin = await qa_agent.create_departure_twin(
            listing_id="lst-agent-compare-001",
            farmer_photos=["s3://bucket/farm.jpg"],
            agent_photos=[],
            quality_result=qr,
        )
        diff = await qa_agent.compare_twin(
            twin_id=twin.twin_id,
            arrival_photos=["s3://bucket/arrival.jpg"],
        )
        assert isinstance(diff, DiffReport)
        assert diff.grade_departure == "A"
        assert diff.liability in ("farmer", "hauler", "buyer", "shared", "none")

    @pytest.mark.asyncio
    async def test_compare_twin_unknown_id_raises(
        self, qa_agent: QualityAssessmentAgent
    ):
        """Agent.compare_twin() raises ValueError for unknown twin_id."""
        with pytest.raises(ValueError):
            await qa_agent.compare_twin(
                twin_id="dt-does-not-exist",
                arrival_photos=["s3://bucket/img.jpg"],
            )


# * ═══════════════════════════════════════════════════════════════
# * EXTENDED TESTS (TASK 17) — Parametrized
# * ═══════════════════════════════════════════════════════════════

class TestArrivalGradeEstimationParametrized:
    """Parametrized grade degradation based on transit/shelf-life ratio."""

    # * Engine: <10% no drop; <25% 1 drop; <50% 1 drop if A+/A else 2; else 2 drops
    @pytest.mark.parametrize("degradation_ratio, dep_grade, expected_arrival_grade", [
        (0.05, "A+", "A+"),
        (0.05, "B", "B"),
        (0.15, "A+", "A"),
        (0.15, "A", "B"),
        (0.30, "A+", "A"),   # 1 drop: A+ → A
        (0.30, "A", "B"),    # 1 drop: A → B
        (0.60, "A+", "B"),   # 2 drops: A+ → B (capped at idx 2)
        (0.60, "A", "C"),    # 2 drops: A → C
    ])
    def test_estimate_arrival_grade_various_ratios(
        self, degradation_ratio, dep_grade, expected_arrival_grade
    ):
        """Test degradation logic based on ratio thresholds (0.1, 0.25, 0.50)."""
        twin = DigitalTwin(
            twin_id="dt-param",
            listing_id="lst",
            farmer_photos=[],
            agent_photos=[],
            grade=dep_grade,
            confidence=0.8,
            defect_types=[],
            defect_count=0,
            shelf_life_days=5,
            gps_lat=12.0,
            gps_lng=77.0,
            ai_annotations={},
            created_at=datetime.now(),
        )
        shelf_life_hours = 5 * 24.0
        transit_hours = shelf_life_hours * degradation_ratio
        grade = _estimate_arrival_grade(
            twin, ["s3://arrival.jpg"], transit_hours
        )
        assert grade == expected_arrival_grade
