"""
Test: Agentic RAG Orchestrator — Mock Mode
==========================================
Validates the orchestrator logic with mock tools and LLMs.
No external services or API keys required.

Usage:
    uv run python scripts/test_agentic_rag.py --mock
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Mock Implementations ──────────────────────────────────────────────────────

class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockPlannerLLM:
    """Returns deterministic plans for test queries."""

    PLANS = {
        "simple": '{"plan": [{"tool_name": "vector_search", "params": {"query": "tomato growing guide"}, "can_parallelize": true, "priority": 1}], "confidence_threshold": 0.70, "plan_reasoning": "Simple agronomy query"}',
        "complex": '{"plan": [{"tool_name": "price_api", "params": {"commodity": "tomato", "location": "Hubli"}, "can_parallelize": true, "priority": 1}, {"tool_name": "vector_search", "params": {"query": "tomato storage strategies"}, "can_parallelize": true, "priority": 1}], "confidence_threshold": 0.80, "plan_reasoning": "Price + market decision query"}',
    }

    def __init__(self, plan_type: str = "simple"):
        self._plan_type = plan_type

    async def generate(self, messages, **kwargs) -> MockLLMResponse:
        return MockLLMResponse(self.PLANS[self._plan_type])


class MockDrafterLLM:
    """Returns simple deterministic drafts."""

    async def generate(self, messages, **kwargs) -> MockLLMResponse:
        return MockLLMResponse(
            "Based on the retrieved information, achi kheti ke liye aapko "
            "proper seedbed preparation karni chahiye. Karnataka mein tomato July "
            "mein lagane se best results milte hain."
        )


class MockVerifierLLM:
    """Always selects draft 0."""

    async def generate(self, messages, **kwargs) -> MockLLMResponse:
        return MockLLMResponse('{"best_draft_index": 0, "reason": "Most detailed and accurate"}')


class MockEvaluatorLLM:
    """Returns high confidence for tests."""

    RESPONSES = {
        "pass":  '{"faithfulness": 0.88, "relevance": 0.90, "reasoning": "Answer well-grounded in context"}',
        "fail":  '{"faithfulness": 0.45, "relevance": 0.50, "reasoning": "Answer lacks specifics from context"}',
    }

    def __init__(self, mode: str = "pass"):
        self._mode = mode

    async def generate(self, messages, **kwargs) -> MockLLMResponse:
        return MockLLMResponse(self.RESPONSES[self._mode])


class MockKnowledgeBase:
    """Returns 5 fake documents."""

    async def search(self, query: str, top_k: int = 5):
        from types import SimpleNamespace
        docs = [
            SimpleNamespace(
                text=f"Document {i} about {query}: Karnataka farmers use drip irrigation for tomato cultivation in Kolar district. Rabi season tomatoes yield 25-30 tonnes per hectare with proper NPK fertilization.",
                id=f"doc_{i}",
                score=0.85 - i * 0.02,
                metadata={"source": "knowledge_base"},
            )
            for i in range(top_k)
        ]
        return SimpleNamespace(documents=docs)


# ── Test Functions ────────────────────────────────────────────────────────────

async def test_simple_orchestration():
    """Test a simple agronomy query — single vector_search tool."""
    print("\n" + "─" * 60)
    print("Test 1: Simple agronomy query orchestration")
    print("─" * 60)

    from src.rag.agentic_orchestrator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator(
        planner_llm=MockPlannerLLM("simple"),
        drafter_llm=MockDrafterLLM(),
        verifier_llm=MockVerifierLLM(),
        evaluator_llm=MockEvaluatorLLM("pass"),
        knowledge_base=MockKnowledgeBase(),
    )

    result = await orchestrator.orchestrate(
        query="How do I grow tomatoes in Karnataka during kharif season?",
        has_image=False,
    )

    print(f"  Answer length: {len(result.answer)} chars")
    print(f"  Tools called: {result.tools_called}")
    print(f"  Retry count: {result.retry_count}")
    print(f"  Latency: {result.total_latency_ms:.0f}ms")
    print(f"  Confidence: {result.eval_gate.confidence:.2f}" if result.eval_gate else "  No eval gate")
    print(f"  Answer: {result.answer[:100]}...")

    assert result.answer, "Expected non-empty answer"
    assert result.retry_count == 0, f"Expected 0 retries, got {result.retry_count}"
    assert "vector_search" in result.tools_called, "Expected vector_search in tools_called"
    print("  ✅ PASS")
    return True


async def test_complex_orchestration():
    """Test a complex decision query — multiple parallel tools."""
    print("\n" + "─" * 60)
    print("Test 2: Complex decision query (price + knowledge plan)")
    print("─" * 60)

    from src.rag.agentic_orchestrator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator(
        planner_llm=MockPlannerLLM("complex"),
        drafter_llm=MockDrafterLLM(),
        verifier_llm=MockVerifierLLM(),
        evaluator_llm=MockEvaluatorLLM("pass"),
        knowledge_base=MockKnowledgeBase(),
        price_client=None,  # price_api tool will log warning and return []
    )

    result = await orchestrator.orchestrate(
        query="Should I sell my tomatoes now or store them for 2 weeks?",
        has_image=False,
    )

    print(f"  Tools in plan: {result.tools_called}")
    print(f"  Documents retrieved: {len(result.retrieved_documents)}")
    print(f"  Confidence: {result.eval_gate.confidence:.2f}" if result.eval_gate else "  No eval gate")

    assert len(result.tools_called) == 2, f"Expected 2 tools, got {result.tools_called}"
    assert "price_api" in result.tools_called
    assert "vector_search" in result.tools_called
    print("  ✅ PASS")
    return True


async def test_self_evaluator_retry():
    """Test that low-confidence answers trigger retry."""
    print("\n" + "─" * 60)
    print("Test 3: Self-evaluator low confidence → retry flag")
    print("─" * 60)

    from types import SimpleNamespace

    from src.rag.agentic_orchestrator import AgenticSelfEvaluator

    evaluator = AgenticSelfEvaluator(llm=MockEvaluatorLLM("fail"))

    docs = [
        SimpleNamespace(
            text="Karnataka tomato market: prices are volatile in December due to holiday demand.",
            id="doc1",
            score=0.8,
        )
    ]

    gate = await evaluator.evaluate(
        query="Should I sell tomatoes now?",
        answer="Yes, sell them.",
        retrieved_docs=docs,
        confidence_threshold=0.75,
    )

    print(f"  Faithfulness: {gate.faithfulness:.2f}")
    print(f"  Relevance: {gate.relevance:.2f}")
    print(f"  Confidence: {gate.confidence:.2f}")
    print(f"  Should retry: {gate.should_retry}")

    assert gate.should_retry, f"Expected retry=True (conf={gate.confidence:.2f} < 0.75)"
    assert gate.confidence < 0.75, f"Expected conf < 0.75, got {gate.confidence:.2f}"
    print("  ✅ PASS — low confidence correctly triggers retry flag")
    return True


async def test_speculative_drafts():
    """Test parallel draft generation with 3 subsets."""
    print("\n" + "─" * 60)
    print("Test 4: Speculative draft engine — 3 parallel drafts")
    print("─" * 60)

    from types import SimpleNamespace

    from src.rag.agentic_orchestrator import SpeculativeDraftEngine

    engine = SpeculativeDraftEngine(
        drafter_llm=MockDrafterLLM(),
        verifier_llm=MockVerifierLLM(),
        n_subsets=3,
    )

    # Create 9 fake documents (3 per subset)
    docs = [
        SimpleNamespace(
            text=f"Agricultural knowledge chunk {i}: Tomatoes need well-drained soil and full sun exposure.",
            id=f"doc_{i}",
            score=0.8,
        )
        for i in range(9)
    ]

    best_answer, best_idx = await engine.generate_and_select(docs, "How to grow tomatoes?")

    print(f"  Best draft index: {best_idx}")
    print(f"  Answer length: {len(best_answer)} chars")
    print(f"  Answer: {best_answer[:80]}...")

    assert best_answer, "Expected non-empty answer"
    assert best_idx == 0, f"MockVerifierLLM should always pick draft 0, got {best_idx}"
    print("  ✅ PASS")
    return True


async def test_retrieval_planner():
    """Test the retrieval planner JSON parsing."""
    print("\n" + "─" * 60)
    print("Test 5: RetrievalPlanner — JSON plan parsing")
    print("─" * 60)

    from src.rag.agentic_orchestrator import RetrievalPlanner

    planner = RetrievalPlanner(llm=MockPlannerLLM("complex"))
    plan = await planner.plan("Should I sell tomatoes now in Hubli?")

    print(f"  Plan steps: {len(plan.plan)}")
    for step in plan.plan:
        print(f"    - {step.tool_name} | parallel={step.can_parallelize}")
    print(f"  Confidence threshold: {plan.confidence_threshold}")
    print(f"  Plan reasoning: {plan.plan_reasoning}")

    assert len(plan.plan) == 2, f"Expected 2 tool calls, got {len(plan.plan)}"
    assert plan.plan[0].tool_name == "price_api"
    assert plan.plan[1].tool_name == "vector_search"
    assert all(t.can_parallelize for t in plan.plan), "Both should be parallelizable"
    print("  ✅ PASS")
    return True


async def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("AGENTIC RAG ORCHESTRATOR — Mock Integration Tests")
    print("=" * 70)

    tests = [
        ("Simple orchestration", test_simple_orchestration),
        ("Complex decision query", test_complex_orchestration),
        ("Self-evaluator low confidence", test_self_evaluator_retry),
        ("Speculative draft engine", test_speculative_drafts),
        ("Retrieval planner parsing", test_retrieval_planner),
    ]

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        try:
            success = await test_fn()
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  ❌ EXCEPTION in '{test_name}': {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: ✅ {passed} passed | ❌ {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    if "--mock" not in sys.argv:
        print("Usage: uv run python scripts/test_agentic_rag.py --mock")
        sys.exit(1)

    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
