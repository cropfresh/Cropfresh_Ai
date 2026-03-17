from src.agents.tool_registry_setup import build_tool_registry
from src.rag.agentic.planner import RetrievalPlanner


def test_build_tool_registry_includes_rate_tools() -> None:
    registry = build_tool_registry()
    tools = set(registry.list_tools())
    assert "multi_source_rates" in tools
    assert "price_api" in tools
    assert "agmarknet" in tools
    assert "deep_research" in tools


def test_planner_fallback_uses_multi_source_rates_for_price_queries() -> None:
    planner = RetrievalPlanner(llm=None)
    plan = planner._fallback_plan("What is the tomato mandi price in Kolar today?")
    assert plan.plan[0].tool_name == "multi_source_rates"
    assert plan.plan[0].params["rate_kinds"] == ["mandi_wholesale"]


def test_planner_fallback_uses_multi_source_rates_for_gold_queries() -> None:
    planner = RetrievalPlanner(llm=None)
    plan = planner._fallback_plan("Tell me the gold rate in Karnataka")
    assert plan.plan[0].tool_name == "multi_source_rates"
    assert plan.plan[0].params["rate_kinds"] == ["gold"]
