"""
LLM Provider Factory
====================
Factory for creating the appropriate LLM provider.
"""

from .base import BaseLLMProvider
from .bedrock import BedrockProvider
from .groq import GroqProvider
from .together import TogetherProvider
from .vllm import VLLMProvider


def create_llm_provider(
    provider: str,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    region: str = "ap-south-1",
    aws_profile: str = "",
) -> BaseLLMProvider:
    """
    Factory function to create the appropriate LLM provider.
    """
    if provider == "bedrock":
        return BedrockProvider(
            model=model or "claude-sonnet",
            region=region,
            profile=aws_profile,
        )
    elif provider == "groq":
        return GroqProvider(
            api_key=api_key,
            model=model or "llama-3.3-70b-versatile",
        )
    elif provider == "together":
        return TogetherProvider(
            api_key=api_key,
            model=model or "meta-llama/Llama-3.3-70B-Instruct",
        )
    elif provider == "vllm":
        return VLLMProvider(
            base_url=base_url,
            model=model or "llama-3.3-70b",
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Use 'bedrock', 'groq', 'together', or 'vllm'."
        )
