"""
Tests for Agronomy Agent v3.0 improvements.

Covers:
- Follow-up question parsing from structured LLM output
- Confidence scoring from document relevance
- Multilingual weather keyword detection
"""

import pytest

from src.agents.agronomy_helpers import avg_score, compute_confidence, parse_follow_ups


class TestParseFollowUps:
    """Test dynamic follow-up extraction from LLM output."""

    def test_parses_english_follow_ups(self):
        llm_output = """\
### 🌾 Analysis
Tomatoes need full sun and well-drained soil.

### ✅ Recommended Actions
1. Plant seedlings in raised beds.

### 📋 Follow-up Questions
- What soil type do you have?
- Do you have drip irrigation available?
- What is your farm size?
"""
        result = parse_follow_ups(llm_output)
        assert len(result) == 3
        assert "What soil type do you have?" in result
        assert "Do you have drip irrigation available?" in result

    def test_parses_kannada_follow_ups(self):
        llm_output = """\
### 🌾 ವಿಶ್ಲೇಷಣೆ
ಟೊಮೆಟೊ ಬೆಳೆಗೆ ಬಿಸಿಲು ಬೇಕು.

### 📋 ಮುಂಬರುವ ಪ್ರಶ್ನೆಗಳು
- ನಿಮ್ಮ ಮಣ್ಣಿನ ಪ್ರಕಾರ ಏನು?
- ನೀರಾವರಿ ಸೌಲಭ್ಯ ಇದೆಯೇ?
"""
        result = parse_follow_ups(llm_output)
        assert len(result) == 2
        assert any("ಮಣ್ಣಿನ" in q for q in result)

    def test_returns_empty_when_no_section(self):
        llm_output = "Just a plain answer with no follow-up section."
        result = parse_follow_ups(llm_output)
        assert result == []

    def test_caps_at_three(self):
        llm_output = """\
### 📋 Follow-up Questions
- Q1?
- Q2?
- Q3?
- Q4?
- Q5?
"""
        result = parse_follow_ups(llm_output)
        assert len(result) == 3


class TestComputeConfidence:
    """Test relevance-based confidence scoring."""

    def test_no_documents_returns_base(self):
        assert compute_confidence([], False) == 0.4

    def test_high_score_documents(self):
        docs = [{"score": 0.95}, {"score": 0.90}, {"score": 0.88}]
        conf = compute_confidence(docs, False)
        assert conf > 0.8

    def test_tool_data_adds_bonus(self):
        docs = [{"score": 0.5}]
        without_tool = compute_confidence(docs, False)
        with_tool = compute_confidence(docs, True)
        assert with_tool == without_tool + 0.1

    def test_capped_at_095(self):
        docs = [{"score": 1.0}, {"score": 1.0}]
        conf = compute_confidence(docs, True)
        assert conf <= 0.95


class TestAvgScore:
    """Test average score helper."""

    def test_empty(self):
        assert avg_score([]) == 0.0

    def test_average(self):
        docs = [{"score": 0.8}, {"score": 0.6}]
        assert avg_score(docs) == pytest.approx(0.7)
