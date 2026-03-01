"""
Test script for LLM Provider
============================
Test both Groq and Amazon Bedrock providers.

Usage:
    python scripts/test_llm.py                  # Test default provider (from .env)
    python scripts/test_llm.py --provider groq  # Test Groq specifically
    python scripts/test_llm.py --provider bedrock  # Test Bedrock specifically
"""

import argparse
import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def check_bedrock_credentials() -> tuple[bool, str]:
    """Validate AWS credentials for Bedrock."""
    try:
        import boto3

        session = boto3.Session(
            region_name=os.getenv("AWS_REGION", "ap-south-1"),
            profile_name=os.getenv("AWS_PROFILE") or None,
        )
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        return True, f"AWS Account: {identity['Account']}, ARN: {identity['Arn']}"
    except Exception as e:
        return False, f"AWS credentials invalid: {e}"


def check_groq_credentials() -> tuple[bool, str]:
    """Validate Groq API key."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        return False, "GROQ_API_KEY not set in .env"
    return True, f"API key present ({api_key[:8]}...)"


async def test_provider(provider: str):
    """Test a specific LLM provider."""
    from orchestrator.llm_provider import LLMMessage, create_llm_provider

    print(f"\n🚀 Testing {provider.upper()} LLM Provider...")
    print("─" * 50)

    # Validate credentials first
    if provider == "bedrock":
        ok, info = check_bedrock_credentials()
        print(f"  🔑 AWS Credentials: {info}")
        if not ok:
            print(f"  ❌ Cannot test Bedrock — fix credentials first.")
            return 1
    elif provider == "groq":
        ok, info = check_groq_credentials()
        print(f"  🔑 Groq API Key: {info}")
        if not ok:
            print(f"  ❌ Cannot test Groq — set GROQ_API_KEY in .env.")
            return 1

    # Create provider
    kwargs: dict = {"provider": provider}
    if provider == "groq":
        kwargs["api_key"] = os.getenv("GROQ_API_KEY", "")
        kwargs["model"] = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    elif provider == "bedrock":
        kwargs["model"] = os.getenv("LLM_MODEL", "claude-sonnet-4")
        kwargs["region"] = os.getenv("AWS_REGION", "ap-south-1")
        kwargs["aws_profile"] = os.getenv("AWS_PROFILE", "")

    try:
        llm = create_llm_provider(**kwargs)
    except Exception as e:
        print(f"  ❌ Provider creation failed: {e}")
        return 1

    # Test generation
    messages = [
        LLMMessage(role="system", content="You are an AI assistant for Indian farmers. Respond concisely."),
        LLMMessage(role="user", content="What is the best time to plant tomatoes in Karnataka?"),
    ]

    print(f"  📤 Sending: 'What is the best time to plant tomatoes in Karnataka?'")

    try:
        response = await llm.generate(messages, temperature=0.7, max_tokens=300)

        print(f"\n  📥 Response:")
        print(f"  {'─' * 46}")
        for line in response.content.split("\n"):
            print(f"  {line}")
        print(f"  {'─' * 46}")
        print(f"\n  📊 Usage:")
        print(f"     Model: {response.model}")
        print(f"     Prompt tokens: {response.usage.get('prompt_tokens', 'N/A')}")
        print(f"     Completion tokens: {response.usage.get('completion_tokens', 'N/A')}")
        print(f"     Total tokens: {response.usage.get('total_tokens', 'N/A')}")
        print(f"\n  ✅ {provider.upper()} provider working correctly!")
        return 0

    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


async def main():
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Test LLM providers for CropFresh AI")
    parser.add_argument(
        "--provider",
        choices=["bedrock", "groq", "together", "all"],
        default=os.getenv("LLM_PROVIDER", "bedrock"),
        help="Provider to test (default: from LLM_PROVIDER env var or 'bedrock')",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  CropFresh AI — LLM Provider Test")
    print("=" * 50)

    if args.provider == "all":
        results = {}
        for p in ["bedrock", "groq"]:
            results[p] = await test_provider(p)
        print("\n" + "=" * 50)
        print("  Summary:")
        for p, r in results.items():
            status = "✅ PASS" if r == 0 else "❌ FAIL"
            print(f"    {p:10s} → {status}")
        return max(results.values())
    else:
        return await test_provider(args.provider)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
