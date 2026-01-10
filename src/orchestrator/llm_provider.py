"""
LLM Provider Abstraction Layer
==============================
Unified interface for switching between LLM providers:
- Groq (Development)
- Together AI (Backup)
- vLLM (Production - Self-hosted)
"""

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
) -> BaseLLMProvider:
    """
    Factory function to create the appropriate LLM provider.
    
    Args:
        provider: "groq", "together", or "vllm"
        api_key: API key for the provider
        base_url: Base URL for vLLM
        model: Model name override
        
    Returns:
        Configured LLM provider instance
    """
    if provider == "groq":
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
        raise ValueError(f"Unknown provider: {provider}. Use 'groq', 'together', or 'vllm'.")
