"""
Unit tests for Quality Assessment Agent — grading logic,
HITL flagging, and rule-based fallback.
"""

import pytest

from src.agents.quality_assessment.agent import (
    HITL_CONFIDENCE_THRESHOLD,
    QualityAssessmentAgent,
)


@pytest.fixture
def agent() -> QualityAssessmentAgent:
    return QualityAssessmentAgent(llm=None)


class TestRuleBasedAssessment:

    @pytest.mark.asyncio
    async def test_positive_description_grades_higher(self, agent: QualityAssessmentAgent):
        """Fresh, firm, clean produce should get A or B grade."""
        report = await agent.assess(
            listing_id="lst-001",
            commodity="Tomato",
            description="Fresh, firm, red, clean, uniform size",
        )

        assert report.assessment.grade in ("A", "B")
        assert report.method == "rule_based"

    @pytest.mark.asyncio
    async def test_negative_description_grades_lower(self, agent: QualityAssessmentAgent):
        """Damaged, soft produce should get C grade."""
        report = await agent.assess(
            listing_id="lst-002",
            commodity="Tomato",
            description="Soft, damaged, brown spots, rotten smell",
        )

        assert report.assessment.grade == "C"

    @pytest.mark.asyncio
    async def test_hitl_always_required_for_rule_based(self, agent: QualityAssessmentAgent):
        """Rule-based assessment always flags HITL review."""
        report = await agent.assess(
            listing_id="lst-003",
            commodity="Onion",
            description="Looks good",
        )

        assert report.assessment.hitl_required is True

    @pytest.mark.asyncio
    async def test_defects_detected_from_keywords(self, agent: QualityAssessmentAgent):
        """Known defect keywords should be extracted."""
        report = await agent.assess(
            listing_id="lst-004",
            commodity="Potato",
            description="Has worm hole and some bruise marks, colour off",
        )

        assert "worm_hole" in report.assessment.defects_detected
        assert "bruise" in report.assessment.defects_detected
        assert "colour_off" in report.assessment.defects_detected

    @pytest.mark.asyncio
    async def test_empty_description_defaults_to_b(self, agent: QualityAssessmentAgent):
        """No description → defaults to B with HITL flag."""
        report = await agent.assess(
            listing_id="lst-005",
            commodity="Beans",
            description="",
        )

        assert report.assessment.grade == "B"
        assert report.assessment.hitl_required is True

    @pytest.mark.asyncio
    async def test_shelf_life_per_commodity(self, agent: QualityAssessmentAgent):
        """Shelf life should vary by commodity."""
        tomato = await agent.assess("lst-t", "Tomato", "fresh")
        onion = await agent.assess("lst-o", "Onion", "fresh")

        assert tomato.assessment.shelf_life_days < onion.assessment.shelf_life_days


class TestProcessViaSupervisor:

    @pytest.mark.asyncio
    async def test_process_without_llm(self, agent: QualityAssessmentAgent):
        """Process should return a helpful text response without LLM."""
        response = await agent.process("How is the quality of my tomatoes?")

        assert response.agent_name == "quality_assessment"
        assert len(response.content) > 0
        assert response.confidence > 0


class TestHITLThreshold:

    def test_threshold_is_ninety_five_percent(self):
        assert HITL_CONFIDENCE_THRESHOLD == 0.95
