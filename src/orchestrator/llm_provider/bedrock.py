"""
Bedrock LLM Provider
====================
Amazon Bedrock provider for production workloads.
"""

import asyncio
from typing import Any, AsyncIterator
from loguru import logger

from .models import LLMMessage, LLMResponse
from .base import BaseLLMProvider


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

        logger.info(
            "BedrockProvider initialized — model={} region={}",
            self.model_id,
            region,
        )

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
