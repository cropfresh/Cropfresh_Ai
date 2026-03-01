"""
LLM Provider Abstraction Layer
==============================
Unified interface for switching between LLM providers:
- Amazon Bedrock (Production — Claude, Llama, Titan)
- Groq (Development — fast inference)
- Together AI (Backup)
- vLLM (Self-hosted)
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel


class LLMMessage(BaseModel):
    """Standard message format."""
    role: str  # system, user, assistant
    content: str


class LLMResponse(BaseModel):
    """Standard response format."""
    content: str
    model: str
    usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM."""
        pass


class BedrockProvider(BaseLLMProvider):
    """
    Amazon Bedrock provider for production workloads.

    Supports Claude (Anthropic), Llama (Meta), and Titan (Amazon) models
    via the Bedrock Converse API for a unified interface.
    """

    MODEL_ALIASES: dict[str, str] = {
        "claude-sonnet-4.6": "global.anthropic.claude-sonnet-4-6",
        "claude-sonnet-4.5": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-sonnet-4": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
        "claude-sonnet": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
        "claude-3.7-sonnet": "apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "claude-3.5-sonnet": "apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-opus": "global.anthropic.claude-opus-4-6-v1",
        "llama-70b": "meta.llama3-70b-instruct-v1:0",
        "llama-8b": "meta.llama3-8b-instruct-v1:0",
        "titan-embed": "amazon.titan-embed-text-v2:0",
    }

    def __init__(
        self,
        model: str = "claude-sonnet",
        region: str = "ap-south-1",
        profile: str = "",
    ):
        import boto3

        session_kwargs: dict[str, Any] = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile

        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime")
        self.model_id = self.MODEL_ALIASES.get(model, model)
        self.region = region

    def _build_converse_params(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the params dict for the Bedrock Converse API."""
        system_parts: list[dict] = []
        converse_messages: list[dict] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append({"text": msg.content})
            else:
                converse_messages.append({
                    "role": msg.role,
                    "content": [{"text": msg.content}],
                })

        params: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }

        if system_parts:
            params["system"] = system_parts

        if "top_p" in kwargs:
            params["inferenceConfig"]["topP"] = kwargs.pop("top_p")

        return params

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate using Amazon Bedrock Converse API."""
        params = self._build_converse_params(
            messages, temperature, max_tokens, **kwargs,
        )

        response = await asyncio.to_thread(
            self._client.converse, **params,
        )

        output_message = response["output"]["message"]
        content_blocks = output_message.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks
        )

        usage_raw = response.get("usage", {})

        return LLMResponse(
            content=text,
            model=self.model_id,
            usage={
                "prompt_tokens": usage_raw.get("inputTokens", 0),
                "completion_tokens": usage_raw.get("outputTokens", 0),
                "total_tokens": usage_raw.get("totalTokens", 0),
            },
            finish_reason=response.get("stopReason", "end_turn"),
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream using Amazon Bedrock ConverseStream API."""
        params = self._build_converse_params(
            messages, temperature, max_tokens, **kwargs,
        )

        response = await asyncio.to_thread(
            self._client.converse_stream, **params,
        )

        event_stream = response.get("stream", [])
        for event in event_stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield text


class GroqProvider(BaseLLMProvider):
    """Groq API provider for development."""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import AsyncGroq
        
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
    
    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate using Groq API."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason or "stop",
        )
    
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream using Groq API."""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class TogetherProvider(BaseLLMProvider):
    """Together AI provider (backup)."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.3-70B-Instruct"):
        import httpx
        
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.together.xyz/v1"
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
    
    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate using Together AI API."""
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [m.model_dump() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )
    
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream using Together AI API."""
        # Implementation for streaming
        raise NotImplementedError("Together streaming not yet implemented")


class VLLMProvider(BaseLLMProvider):
    """vLLM provider for self-hosted production."""
    
    def __init__(self, base_url: str, model: str = "llama-3.3-70b"):
        import httpx
        
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate using vLLM OpenAI-compatible API."""
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [m.model_dump() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )
    
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream using vLLM API."""
        # Implementation for streaming
        raise NotImplementedError("vLLM streaming not yet implemented")


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
    
    Args:
        provider: "bedrock", "groq", "together", or "vllm"
        api_key: API key (for Groq/Together)
        base_url: Base URL (for vLLM)
        model: Model name or Bedrock alias (e.g. "claude-sonnet", "llama-70b")
        region: AWS region (for Bedrock)
        aws_profile: AWS profile name (for Bedrock, optional)
        
    Returns:
        Configured LLM provider instance
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
