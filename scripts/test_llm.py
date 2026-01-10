"""
Test script for LLM Provider
============================
Quick test to verify Groq API connection and LLM generation.

Usage:
    python scripts/test_llm.py
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    from orchestrator.llm_provider import create_llm_provider, LLMMessage
    
    # Get API key
    api_key = os.getenv("GROQ_API_KEY", "")
    
    if not api_key or api_key == "your_groq_api_key_here":
        print("❌ GROQ_API_KEY not set. Please update .env file.")
        return 1
    
    print("🚀 Testing Groq LLM Provider...")
    print("-" * 40)
    
    # Create provider
    provider = create_llm_provider(
        provider="groq",
        api_key=api_key,
    )
    
    # Test generation
    messages = [
        LLMMessage(role="system", content="You are an AI assistant for Indian farmers."),
        LLMMessage(role="user", content="What is the best time to plant tomatoes in Karnataka?"),
    ]
    
    print("📤 Sending query: 'What is the best time to plant tomatoes in Karnataka?'")
    print()
    
    try:
        response = await provider.generate(messages, temperature=0.7, max_tokens=500)
        
        print("📥 Response:")
        print("-" * 40)
        print(response.content)
        print("-" * 40)
        print(f"\n📊 Usage:")
        print(f"   Model: {response.model}")
        print(f"   Prompt tokens: {response.usage.get('prompt_tokens', 'N/A')}")
        print(f"   Completion tokens: {response.usage.get('completion_tokens', 'N/A')}")
        print(f"   Total tokens: {response.usage.get('total_tokens', 'N/A')}")
        print(f"\n✅ Groq provider working correctly!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
