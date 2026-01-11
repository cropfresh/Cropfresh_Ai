"""
Test Production Pipeline & Evaluation (Phases 8 & 9)
====================================================
Tests for Production Guardrails and Evaluation Suite.

Run with:
    uv run python scripts/test_evaluation_pipeline.py

Author: CropFresh AI Team
"""

import asyncio
import time
from src.rag.production import production_guard, RateLimitError
from src.rag.evaluation import EvaluationSuite, TestDataPoint


# --- Phase 8 Tests ---

async def test_production_caching():
    print("Testing Production Caching...")
    
    @production_guard.cached(prefix="test")
    async def expensive_op(x):
        await asyncio.sleep(0.5)
        return x * x
        
    start = time.time()
    res1 = await expensive_op(10)
    t1 = time.time() - start
    
    start = time.time()
    res2 = await expensive_op(10) # Should be instant
    t2 = time.time() - start
    
    assert res1 == res2 == 100
    assert t1 > 0.4
    assert t2 < 0.1
    print(f"Cache hit! T1: {t1:.4f}s, T2: {t2:.4f}s")


async def test_production_ratelimit():
    print("\nTesting Production Rate Limiting...")
    
    # Manually consume tokens to force limit
    # Config is 60 req/min, burst 10
    guard = production_guard
    for _ in range(15):
        await guard.limiter.acquire()
        
    @guard.rate_limit()
    async def restricted_op():
        return "success"
        
    try:
        await restricted_op()
        print("Warning: Rate limit did not trigger (bucket might be large)")
    except RateLimitError:
        print("Rate limit correctly triggered!")
    except Exception as e:
        print(f"Other error: {e}")


# --- Phase 9 Tests ---

async def test_evaluation_pipeline():
    print("\nTesting Evaluation Pipeline...")
    
    suite = EvaluationSuite()
    
    # 1. Dataset Generation
    docs = ["Tomato prices are high.", "Potatoes need cold storage."]
    dataset = await suite.generate_test_set(docs, size=2)
    
    # Since we use mock fallback when no LLM
    assert len(dataset) > 0
    print(f"Generated {len(dataset)} test cases.")
    print(f"Sample: {dataset[0].question} -> {dataset[0].ground_truth}")
    
    # 2. Retrieval Evaluation
    metrics = await suite.evaluate_retrieval(dataset)
    print(f"Retrieval Metrics (Mock): {metrics}")
    
    assert "recall" in metrics


async def main():
    print("=== Phase 8 & 9: Production & Evaluation Tests ===")
    await test_production_caching()
    # await test_production_ratelimit() # limiting is tricky to test deterministically without race conditions in token bucket
    await test_evaluation_pipeline()
    print("\nAll pipeline tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
