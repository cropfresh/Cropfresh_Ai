"""
LLM Routing Test Script
=======================
An automated script to test the SupervisorAgent's LLM routing capabilities
across various multilingual queries. It cleanly outputs the expected vs actual
routing decisions along with confidence and LLM reasoning.

Usage:
    cd d:\\Cropfresh Ai\\Cropfresh_Ai
    uv run python scripts/test_llm_routing.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_llm_routing():
    """Runs a suite of conversational intents against the LLM router."""
    print("=" * 80)
    print("  CropFresh AI - LLM Supervisor Routing Test")
    print("=" * 80)
    
    try:
        from src.config import get_settings
        from src.orchestrator.llm_provider import create_llm_provider
        from src.agents.supervisor import SupervisorAgent
        
        settings = get_settings()
        
        if not settings.has_llm_configured:
            print("[FAIL] No LLM provider configured. Please check .env settings.")
            return False
            
        print(f"🔄 Initializing Supervisor built on: {settings.llm_provider} / {settings.llm_model}...\n")
        
        llm = create_llm_provider(
            provider=settings.llm_provider,
            api_key=settings.groq_api_key or settings.together_api_key,
            base_url=getattr(settings, "vllm_base_url", ""),
            model=settings.llm_model,
        )
        
        supervisor = SupervisorAgent(llm=llm)
        await supervisor.initialize()

        test_cases = [
            {
                "query": "ನನ್ನ ಟೊಮೆಟೊ ಬೆಳೆ ರೋಗದಿಂದ ಕೂಡಿದೆ, ಪರಿಹಾರ ಏನು?",  # Tomato disease
                "expected": "agronomy_agent"
            },
            {
                "query": "ಇಂದು ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ಈರುಳ್ಳಿ ಬೆಲೆ ಎಷ್ಟು?",  # Onion price
                "expected": "web_scraping_agent"
            },
            {
                "query": "ನಾನು CropFresh ಆಪ್ ಅಲ್ಲಿ ಲಾಗಿನ್ ಆಗೋದು ಹೇಗೆ?",  # App login
                "expected": "platform_agent"
            },
            {
                "query": "What crop should I sow this September for maximum demand?", 
                "expected": "adcl_agent"
            },
            {
                "query": "Who will buy my 10 quintals of cabbage?", 
                "expected": "buyer_matching_agent"
            },
            {
                "query": "ನಮಸ್ಕಾರ / Hello", 
                "expected": "general_agent"
            }
        ]
        
        passed = 0
        print(f"{'Query Snippet':<35} | {'Expected':<22} | {'Actual':<22} | {'Result'}")
        print("-" * 80)
        
        for case in test_cases:
            query = case["query"]
            expected = case["expected"]
            
            # Execute LLM Route
            routing = await supervisor.route_query(query)
            
            actual = routing.agent_name
            status = "[PASS]" if actual == expected else "[FAIL]"
            if actual == expected:
                passed += 1
                
            snippet = (query[:32] + "...") if len(query) > 32 else query
            print(f"{snippet:<35} | {expected:<22} | {actual:<22} | {status}")
            if actual != expected:
                print(f"   -> Reasoning given: {routing.reasoning}")
                
        print("-" * 80)
        print(f"Score: {passed}/{len(test_cases)} tests passed.")
        
        return passed == len(test_cases)
        
    except Exception as e:
        print(f"[FAIL] Routing test failed to run: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_llm_routing())
    sys.exit(0 if success else 1)
