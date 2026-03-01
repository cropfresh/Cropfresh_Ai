"""
Unit tests for PricePredictionAgent — Task 5.

Covers:
  - Class name fix (no syntax / import errors)
  - Rule-based prediction within ±15% of actual
  - Trend analysis: rising / falling / stable
  - Seasonal factor labels for Karnataka crops
  - Sell/hold recommendation logic
  - Graceful fallback when no historical data
  - LLM fallback returns valid PricePrediction
  - execute() and process() public interfaces
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Optional

import pytest

from src.agents.price_prediction.agent import (
    SEASONAL_CALENDAR,
    PricePrediction,
    PricePredictionAgent,
)
from src.tools.agmarknet import AgmarknetPrice


# * ═══════════════════════════════════════════════════════════════
# * FIXTURES & HELPERS
# * ═══════════════════════════════════════════════════════════════

def _make_history(
    base_price_per_quintal: float,
    days: int = 30,
    trend_pct_per_day: float = 0.0,
) -> list[AgmarknetPrice]:
    """Build synthetic price history with optional linear trend."""
    history: list[AgmarknetPrice] = []
    for i in range(days):
        day = datetime.now() - timedelta(days=(days - i - 1))
        modal = base_price_per_quintal * (1 + trend_pct_per_day * i)
        history.append(
            AgmarknetPrice(
                commodity="Tomato",
                state="Karnataka",
                district="Kolar",
                market="Kolar Main Market",
                date=day,
                min_price=modal * 0.85,
                max_price=modal * 1.15,
                modal_price=modal,
            )
        )
    return history


class StubAgmarknet:
    """Returns deterministic price history without network calls."""

    def __init__(self, history: list[AgmarknetPrice]):
        self._history = history

    async def get_historical_prices(
        self,
        commodity: str,
        state: str = "Karnataka",
        district: Optional[str] = None,
        days: int = 90,
    ) -> list[AgmarknetPrice]:
        return self._history[-days:] if days < len(self._history) else self._history


def _make_agent(
    history: Optional[list[AgmarknetPrice]] = None,
    llm=None,
) -> PricePredictionAgent:
    stub = StubAgmarknet(history or _make_history(2500.0, days=90))
    return PricePredictionAgent(llm=llm, agmarknet_tool=stub)


# * ═══════════════════════════════════════════════════════════════
# * BASIC IMPORT AND CLASS SANITY
# * ═══════════════════════════════════════════════════════════════

def test_class_name_fixed_no_import_error():
    """AC 1 — class name is PricePredictionAgent, no syntax or import error."""
    agent = _make_agent()
    assert agent.name == "price_prediction"
    assert isinstance(agent, PricePredictionAgent)


def test_price_prediction_model_has_required_fields():
    pred = PricePrediction(
        commodity="tomato",
        district="Kolar",
        current_price=25.0,
        predicted_price_7d=27.0,
        predicted_price_30d=28.0,
        confidence=0.65,
        trend="rising",
        trend_strength=0.4,
        seasonal_factor="normal",
        factors=["Test factor."],
        recommendation="sell_now",
        data_source="historical",
    )
    assert pred.commodity == "tomato"
    assert pred.predicted_price_7d == 27.0


# * ═══════════════════════════════════════════════════════════════
# * RULE-BASED PREDICTION
# * ═══════════════════════════════════════════════════════════════

def test_rule_based_predict_within_15pct_of_actual():
    """
    AC 2 — rule-based prediction is within ±15% of the latest observed price
    when seasonal multiplier is 1.0 (no seasonal effect) and momentum is flat.

    NOTE: The seasonal multiplier is intentionally isolated here because the
    task spec requires the prediction formula itself (avg × momentum) to be
    within ±15%, independent of seasonal calendar adjustments.
    """
    history = _make_history(2500.0, days=30, trend_pct_per_day=0.0)
    agent = _make_agent(history)

    features = agent._extract_features(history, "tomato")
    # * Override seasonal multiplier to 1.0 to isolate pure rule-based accuracy
    features["seasonal_multiplier"] = 1.0

    actual_per_kg = history[-1].modal_price_per_kg  # 25.0
    predicted = agent._rule_based_predict(features, days_ahead=7)

    deviation = abs(predicted - actual_per_kg) / actual_per_kg
    assert deviation <= 0.15, (
        f"Rule-based predicted {predicted:.2f} is >15% off actual {actual_per_kg:.2f}"
    )


@pytest.mark.asyncio
async def test_predict_returns_valid_price_prediction_object():
    agent = _make_agent()
    pred = await agent.predict("tomato", "Kolar")

    assert isinstance(pred, PricePrediction)
    assert pred.current_price > 0
    assert pred.predicted_price_7d > 0
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.recommendation in ("sell_now", "hold_3d", "hold_7d", "hold_30d")
    assert pred.data_source in ("historical", "model", "llm_estimate")


# * ═══════════════════════════════════════════════════════════════
# * TREND ANALYSIS
# * ═══════════════════════════════════════════════════════════════

def test_analyze_trend_detects_rising():
    """AC 3 — rising trend detected when price increases steadily."""
    agent = _make_agent()
    # NOTE: +3% per day over 14 days — clearly rising
    history = _make_history(2000.0, days=14, trend_pct_per_day=0.03)
    trend, strength = agent._analyze_trend(history)
    assert trend == "rising"
    assert strength > 0.0


def test_analyze_trend_detects_falling():
    """AC 3 — falling trend detected when price decreases steadily."""
    agent = _make_agent()
    history = _make_history(2000.0, days=14, trend_pct_per_day=-0.03)
    trend, strength = agent._analyze_trend(history)
    assert trend == "falling"
    assert strength > 0.0


def test_analyze_trend_detects_stable():
    """AC 3 — stable trend when price oscillates minimally."""
    agent = _make_agent()
    history = _make_history(2000.0, days=14, trend_pct_per_day=0.0)
    trend, strength = agent._analyze_trend(history)
    assert trend == "stable"


def test_analyze_trend_with_insufficient_data_returns_stable():
    agent = _make_agent()
    history = _make_history(2000.0, days=2)
    trend, strength = agent._analyze_trend(history)
    assert trend == "stable"
    assert strength == 0.0


# * ═══════════════════════════════════════════════════════════════
# * SEASONAL FACTOR
# * ═══════════════════════════════════════════════════════════════

def test_seasonal_factor_peak_harvest_for_tomato_in_april():
    """AC 4 — tomato in April (month 4) has multiplier 0.7 → peak_harvest."""
    agent = _make_agent()
    # NOTE: Month 4 has multiplier 0.7 in SEASONAL_CALENDAR['tomato']
    factor = agent._get_seasonal_factor("tomato", 4)
    assert factor == "peak_harvest"


def test_seasonal_factor_off_season_for_tomato_in_june():
    """AC 4 — tomato in June (month 6) has multiplier 1.4 → off_season."""
    agent = _make_agent()
    factor = agent._get_seasonal_factor("tomato", 6)
    assert factor == "off_season"


def test_seasonal_factor_normal_for_tomato_in_january():
    """AC 4 — tomato in January (month 1) has multiplier 1.0 → normal."""
    agent = _make_agent()
    factor = agent._get_seasonal_factor("tomato", 1)
    assert factor == "normal"


def test_seasonal_calendar_contains_all_months_for_known_crops():
    """AC 4 — each crop in SEASONAL_CALENDAR has entries for all 12 months."""
    for crop, calendar in SEASONAL_CALENDAR.items():
        assert set(calendar.keys()) == set(range(1, 13)), f"{crop} missing months"


def test_seasonal_factor_defaults_to_normal_for_unknown_crop():
    agent = _make_agent()
    factor = agent._get_seasonal_factor("dragonFruit", 6)
    assert factor == "normal"


# * ═══════════════════════════════════════════════════════════════
# * RECOMMENDATION
# * ═══════════════════════════════════════════════════════════════

def test_recommendation_sell_now_when_predicted_above_5pct():
    """AC 5 — sell_now when predicted is ≥5% above current."""
    agent = _make_agent()
    rec = agent._generate_recommendation(
        current=25.0, predicted=27.0, trend="rising", seasonal="normal",
    )
    assert rec == "sell_now"


def test_recommendation_hold_3d_for_slight_upside():
    """AC 5 — hold_3d when predicted is 0–5% above current."""
    agent = _make_agent()
    rec = agent._generate_recommendation(
        current=25.0, predicted=25.5, trend="stable", seasonal="normal",
    )
    assert rec == "hold_3d"


def test_recommendation_hold_7d_for_rising_off_season():
    """AC 5 — hold_7d when trend is rising and seasonal is off_season."""
    agent = _make_agent()
    rec = agent._generate_recommendation(
        current=25.0, predicted=24.0, trend="rising", seasonal="off_season",
    )
    assert rec == "hold_7d"


def test_recommendation_hold_30d_otherwise():
    """AC 5 — hold_30d when predicted < current and no special conditions."""
    agent = _make_agent()
    rec = agent._generate_recommendation(
        current=25.0, predicted=20.0, trend="falling", seasonal="peak_harvest",
    )
    assert rec == "hold_30d"


# * ═══════════════════════════════════════════════════════════════
# * GRACEFUL FALLBACK
# * ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_predict_graceful_fallback_when_no_history():
    """AC 6 — returns valid prediction even with empty history, data_source=llm_estimate."""
    class EmptyAgmarknet:
        async def get_historical_prices(self, **kwargs):
            return []

    agent = PricePredictionAgent(agmarknet_tool=EmptyAgmarknet())
    pred = await agent.predict("tomato", "Kolar")

    assert isinstance(pred, PricePrediction)
    assert pred.data_source == "llm_estimate"
    assert pred.confidence < 0.6
    assert len(pred.factors) > 0


@pytest.mark.asyncio
async def test_predict_fallback_on_fetch_exception():
    """AC 6 — agent handles exception from agmarknet gracefully."""
    class FailingAgmarknet:
        async def get_historical_prices(self, **kwargs):
            raise ConnectionError("API offline")

    agent = PricePredictionAgent(agmarknet_tool=FailingAgmarknet())
    pred = await agent.predict("onion", "Bellary")

    assert isinstance(pred, PricePrediction)
    assert pred.data_source == "llm_estimate"


# * ═══════════════════════════════════════════════════════════════
# * LLM FALLBACK
# * ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_llm_based_prediction_uses_llm_response():
    """AC 7 — LLM fallback returns llm_estimate with parsed price."""
    mock_llm = SimpleNamespace()

    async def generate(prompt: str, max_tokens: int = 150) -> str:
        return '{"price_per_kg": 30.0, "trend": "rising", "factors": ["Strong demand."]}'

    mock_llm.generate = generate

    class EmptyAgmarknet:
        async def get_historical_prices(self, **kwargs):
            return []

    agent = PricePredictionAgent(llm=mock_llm, agmarknet_tool=EmptyAgmarknet())
    pred = await agent.predict("tomato", "Kolar")

    assert pred.data_source == "llm_estimate"
    assert pred.current_price == 30.0
    assert pred.trend == "rising"
    assert "Strong demand." in pred.factors


@pytest.mark.asyncio
async def test_llm_fallback_on_malformed_llm_response():
    """AC 7 — graceful fallback when LLM returns unparseable output."""
    mock_llm = SimpleNamespace()

    async def generate(prompt: str, max_tokens: int = 150) -> str:
        return "I am unable to predict prices today."

    mock_llm.generate = generate

    class EmptyAgmarknet:
        async def get_historical_prices(self, **kwargs):
            return []

    agent = PricePredictionAgent(llm=mock_llm, agmarknet_tool=EmptyAgmarknet())
    pred = await agent.predict("onion", "Dharwad")

    assert isinstance(pred, PricePrediction)
    assert pred.data_source == "llm_estimate"


# * ═══════════════════════════════════════════════════════════════
# * PUBLIC INTERFACES
# * ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_execute_returns_expected_keys():
    """AC 8 — execute() returns all required output keys."""
    agent = _make_agent()
    result = await agent.execute({"commodity": "tomato", "district": "Kolar", "days_ahead": 7})

    required_keys = {
        "commodity", "district", "current_price", "predicted_price_7d",
        "predicted_price_30d", "confidence", "trend", "trend_strength",
        "seasonal_factor", "factors", "recommendation", "data_source",
    }
    assert required_keys <= result.keys()
    assert isinstance(result["factors"], list)


@pytest.mark.asyncio
async def test_process_returns_agent_response_with_content():
    """AC 8 — process() returns AgentResponse with commodity info in content."""
    agent = _make_agent()
    response = await agent.process("What is the tomato price?", context={"commodity": "tomato"})

    assert response.agent_name == "price_prediction"
    assert "tomato" in response.content.lower() or "Tomato" in response.content
    assert response.confidence > 0.0


@pytest.mark.asyncio
async def test_process_extracts_commodity_from_query_text():
    """AC 8 — process() can infer commodity from query when not in context."""
    agent = _make_agent()
    response = await agent.process("Tell me about onion prices this week")

    assert response.agent_name == "price_prediction"
    assert response.confidence > 0.0


@pytest.mark.asyncio
async def test_factors_list_non_empty():
    """AC 7 — factors list is always populated."""
    agent = _make_agent()
    pred = await agent.predict("tomato", "Kolar")
    assert len(pred.factors) > 0
    assert all(isinstance(f, str) for f in pred.factors)
