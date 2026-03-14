"""
vLLM Provider
=============
vLLM provider for self-hosted production endpoints.
"""

import json
from typing import Any, AsyncIterator

from .base import BaseLLMProvider
from .models import LLMMessage, LLMResponse


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
        request_data = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=request_data,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        choices = data.get("choices", [])
                        if choices and "delta" in choices[0]:
                            content = choices[0]["delta"].get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
