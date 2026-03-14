"""
Together AI Provider
====================
Together AI provider for backup access.
"""

from typing import Any, AsyncIterator

from .base import BaseLLMProvider
from .models import LLMMessage, LLMResponse


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
        raise NotImplementedError("Together streaming not yet implemented")
