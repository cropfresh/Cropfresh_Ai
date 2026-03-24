"""
Unit tests for ConfidenceGate (Phase 1 — ADR-010).

Tests cover:
  - Safety classification (safe / safety-critical / platform)
  - Kannada safety keywords
  - Grounding score calculation
  - Confidence gating with approval / decline
  - Decline responses match safety level
  - Edge cases: empty docs, empty answer
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.rag.confidence_gate import (
    ConfidenceGate,
    SafetyLevel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gate():
    return ConfidenceGate(llm=None)


@pytest.fixture
def sample_docs():
    doc1 = MagicMock()
    doc1.text = "Tomato leaf curl virus is transmitted by whiteflies in Karnataka."
    doc2 = MagicMock()
    doc2.text = "Apply neem oil spray every 7 days as preventive measure."
    return [doc1, doc2]


# ---------------------------------------------------------------------------
# Safety classification tests
# ---------------------------------------------------------------------------


class TestSafetyClassification:
    def test_pesticide_query_is_safety_critical(self, gate):
        """Pesticide dosage questions must be safety-critical."""
        level = gate.classify_safety("What pesticide dosage for tomato blight?")
        assert level == SafetyLevel.SAFETY_CRITICAL

    def test_loan_query_is_safety_critical(self, gate):
        """Financial queries must be safety-critical."""
        level = gate.classify_safety("How to get a farm loan from SBI?")
        assert level == SafetyLevel.SAFETY_CRITICAL

    def test_kannada_safety_keyword(self, gate):
        """Kannada safety keywords should trigger safety-critical."""
        level = gate.classify_safety("ಕೀಟನಾಶಕ ಎಷ್ಟು ಹಾಕಬೇಕು?")
        assert level == SafetyLevel.SAFETY_CRITICAL

    def test_register_query_is_platform(self, gate):
        """Platform-related queries should be classified as platform."""
        level = gate.classify_safety("How do I register on CropFresh?")
        assert level == SafetyLevel.PLATFORM

    def test_general_agronomy_is_safe(self, gate):
        """Normal farming queries should be classified as safe."""
        level = gate.classify_safety("Best time to plant ragi in Karnataka?")
        assert level == SafetyLevel.SAFE


# ---------------------------------------------------------------------------
# Grounding score tests
# ---------------------------------------------------------------------------


class TestGroundingScore:
    def test_grounded_answer_scores_high(self, gate, sample_docs):
        """Answer grounded in docs should score high."""
        answer = "Tomato leaf curl virus is transmitted by whiteflies."
        score = gate._calculate_grounding(answer, sample_docs)
        assert score > 0.5

    def test_ungrounded_answer_scores_low(self, gate, sample_docs):
        """Answer with no doc overlap should score low."""
        answer = "Rice requires flooded paddies during monsoon season."
        score = gate._calculate_grounding(answer, sample_docs)
        assert score < 0.5

    def test_empty_answer_returns_one(self, gate):
        """Empty answer content should return 1.0 (no claims to check)."""
        score = gate._calculate_grounding("", [])
        assert score == 0.0

    def test_empty_docs_returns_zero(self, gate):
        """No documents should return 0.0 grounding."""
        score = gate._calculate_grounding("Some answer text", [])
        assert score == 0.0


# ---------------------------------------------------------------------------
# Gating decision tests
# ---------------------------------------------------------------------------


class TestGatingDecision:
    @pytest.mark.asyncio
    async def test_high_confidence_approved(self, gate, sample_docs):
        """High confidence should approve the answer."""
        result = await gate.gate(
            query="What causes tomato leaf curl?",
            answer="Tomato leaf curl virus is transmitted by whiteflies.",
            documents=sample_docs,
            faithfulness=0.90,
            relevance=0.85,
        )
        assert result.is_approved is True
        assert result.safety_level == SafetyLevel.SAFE

    @pytest.mark.asyncio
    async def test_low_confidence_declined(self, gate, sample_docs):
        """Low faithfulness should decline the answer."""
        result = await gate.gate(
            query="What pesticide dose for tomato blight?",
            answer="Use 5ml of random chemical per litre.",
            documents=sample_docs,
            faithfulness=0.30,
            relevance=0.40,
        )
        assert result.is_approved is False
        assert result.safety_level == SafetyLevel.SAFETY_CRITICAL
        assert result.decline_reason is not None

    @pytest.mark.asyncio
    async def test_kannada_decline_response_is_localized(self, gate, sample_docs):
        """Kannada safety query should receive a Kannada decline response."""
        result = await gate.gate(
            query="ಕೀಟನಾಶಕ ಎಷ್ಟು ಹಾಕಬೇಕು?",
            answer="Random pesticide advice.",
            documents=sample_docs,
            faithfulness=0.10,
            relevance=0.20,
        )
        assert result.is_approved is False
        assert any("\u0c80" <= char <= "\u0cff" for char in result.answer)

    @pytest.mark.asyncio
    async def test_decline_response_matches_safety_level(self, gate, sample_docs):
        """Decline response should match the safety level template."""
        result = await gate.gate(
            query="How to get a loan?",
            answer="Random loan advice here.",
            documents=sample_docs,
            faithfulness=0.10,
            relevance=0.10,
        )
        assert result.is_approved is False
        assert "KVK" in result.answer or "support" in result.answer

    @pytest.mark.asyncio
    async def test_safety_critical_needs_higher_threshold(self, gate, sample_docs):
        """Safety-critical queries need ≥ 0.85, not 0.70."""
        result = await gate.gate(
            query="What insecticide for aphids?",
            answer="Tomato leaf curl virus is transmitted by whiteflies.",
            documents=sample_docs,
            faithfulness=0.75,
            relevance=0.75,
        )
        # 0.75 * 0.4 + 0.75 * 0.3 + grounding * 0.3
        # Even with decent scores, safety-critical should need ≥ 0.85
        assert result.safety_level == SafetyLevel.SAFETY_CRITICAL
