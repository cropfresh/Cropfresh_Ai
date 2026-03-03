"""
Amazon Bedrock Streaming LLM Module for CropFresh Voice Agent
==============================================================

Provides real-time streaming chat completions via Amazon Bedrock.
Acts as a fallback when Groq is unavailable. Same interface as
GroqLLMStreaming for seamless provider switching.

Features:
- Token-by-token streaming via Bedrock Converse Stream API
- Sentence boundary detection for speculative TTS
- Cancellation support for barge-in interruption
- Latency-optimized inference mode

Usage:
    streamer = BedrockLLMStreaming()
    async for sentence in streamer.stream_sentences("What is wheat price?"):
        await tts.synthesize_stream(sentence.text)
"""

import asyncio
import os
import re
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from loguru import logger

from .groq_streaming import (
    CROPFRESH_SYSTEM_PROMPT,
    SENTENCE_BOUNDARY_RE,
    SentenceChunk,
    StreamingConfig,
)


# ═══════════════════════════════════════════════════════════════
# Bedrock Configuration
# ═══════════════════════════════════════════════════════════════

# Bedrock model IDs
BEDROCK_MODELS = {
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514",
    "claude-3.5-haiku": "anthropic.claude-3-5-haiku-20241022",
    "llama-3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
    "llama-3.1-8b": "meta.llama3-1-8b-instruct-v1:0",
}

DEFAULT_BEDROCK_MODEL = "claude-sonnet-4"


# ═══════════════════════════════════════════════════════════════
# Bedrock Streaming LLM
# ═══════════════════════════════════════════════════════════════

class BedrockLLMStreaming:
    """
    Streaming LLM wrapper for Amazon Bedrock.

    Uses the Bedrock Converse Stream API for real-time token
    streaming. Same interface as GroqLLMStreaming so the duplex
    pipeline can swap providers transparently.

    Usage:
        streamer = BedrockLLMStreaming()
        async for chunk in streamer.stream_sentences("Hello"):
            print(chunk.text)
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        model_name: str = DEFAULT_BEDROCK_MODEL,
        region: Optional[str] = None,
        latency_optimized: bool = True,
    ) -> None:
        self._config = config or StreamingConfig()
        self._model_name = model_name
        self._model_id = BEDROCK_MODELS.get(model_name, model_name)
        self._region = region or os.getenv("AWS_REGION", "ap-south-1")
        self._latency_optimized = latency_optimized
        self._client = None
        self._cancelled = False

        logger.info(
            f"[BedrockStreaming] Configured: model={self._model_id}, "
            f"region={self._region}, latency_opt={latency_optimized}"
        )

    def _ensure_client(self) -> None:
        """Lazy-init the Bedrock Runtime client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self._region,
                )
                logger.info("[BedrockStreaming] Client initialized")
            except ImportError:
                raise ImportError(
                    "boto3 not installed. Run: uv pip install boto3"
                )

    def cancel(self) -> None:
        """Cancel the current stream (for barge-in)."""
        self._cancelled = True
        logger.info("[BedrockStreaming] Stream cancelled (barge-in)")

    def reset(self) -> None:
        """Reset cancellation state for a new turn."""
        self._cancelled = False

    # ──────────────────────────────────────────────────────────
    # Token-level streaming
    # ──────────────────────────────────────────────────────────

    async def stream_tokens(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        language: str = "en",
    ) -> AsyncIterator[str]:
        """
        Stream individual tokens from Bedrock.

        Uses run_in_executor since boto3 is synchronous.

        Yields:
            Individual tokens as strings.
        """
        self._ensure_client()
        self.reset()

        messages = self._build_converse_messages(
            user_message, conversation_history, language
        )
        system_prompt = self._get_system_prompt(language)

        loop = asyncio.get_event_loop()

        # Build the API kwargs
        kwargs = {
            "modelId": self._model_id,
            "messages": messages,
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "temperature": self._config.temperature,
                "maxTokens": self._config.max_tokens,
            },
        }

        # Add latency optimization if supported
        if self._latency_optimized and "anthropic" in self._model_id:
            kwargs["performanceConfig"] = {"latency": "optimized"}

        try:
            # Bedrock converse_stream is synchronous — run in executor
            response = await loop.run_in_executor(
                None,
                lambda: self._client.converse_stream(**kwargs),
            )

            stream = response.get("stream", [])
            for event in stream:
                if self._cancelled:
                    logger.info("[BedrockStreaming] Token stream cancelled")
                    break

                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        yield text

        except Exception as e:
            logger.error(f"[BedrockStreaming] Stream error: {e}")
            raise

    # ──────────────────────────────────────────────────────────
    # Sentence-level streaming (for TTS pipeline)
    # ──────────────────────────────────────────────────────────

    async def stream_sentences(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        language: str = "en",
    ) -> AsyncIterator[SentenceChunk]:
        """
        Stream complete sentences from Bedrock.

        Same interface as GroqLLMStreaming.stream_sentences().
        """
        buffer = ""
        token_count = 0

        async for token in self.stream_tokens(
            user_message, conversation_history, language
        ):
            if self._cancelled:
                break

            buffer += token
            token_count += 1

            # Check for sentence boundaries
            sentences = SENTENCE_BOUNDARY_RE.split(buffer)

            if len(sentences) > 1:
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        yield SentenceChunk(
                            text=sentence,
                            is_final=False,
                            token_count=token_count,
                        )
                buffer = sentences[-1]

        # Flush remaining text
        if buffer.strip() and not self._cancelled:
            yield SentenceChunk(
                text=buffer.strip(),
                is_final=True,
                token_count=token_count,
            )

    # ──────────────────────────────────────────────────────────
    # Non-streaming
    # ──────────────────────────────────────────────────────────

    async def generate(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        language: str = "en",
    ) -> str:
        """Generate a complete response (non-streaming)."""
        tokens = []
        async for token in self.stream_tokens(
            user_message, conversation_history, language
        ):
            tokens.append(token)
        return "".join(tokens)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _get_system_prompt(self, language: str) -> str:
        """Build language-adapted system prompt."""
        prompt = self._config.system_prompt
        if language != "en":
            lang_names = {
                "hi": "Hindi", "kn": "Kannada", "te": "Telugu",
                "ta": "Tamil", "ml": "Malayalam", "mr": "Marathi",
                "gu": "Gujarati", "bn": "Bengali", "pa": "Punjabi",
                "or": "Odia",
            }
            lang_name = lang_names.get(language, language)
            prompt += (
                f"\n\nIMPORTANT: The user is speaking in {lang_name}. "
                f"You MUST respond in {lang_name}."
            )
        return prompt

    def _build_converse_messages(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]],
        language: str,
    ) -> list[dict]:
        """Build Bedrock Converse API message format."""
        messages = []

        # Add conversation history (limit to last 6)
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = msg.get("role", "user")
                # Bedrock uses "user" and "assistant" roles only
                if role in ("user", "assistant"):
                    messages.append({
                        "role": role,
                        "content": [{"text": msg.get("content", "")}],
                    })

        # Add current user message
        messages.append({
            "role": "user",
            "content": [{"text": user_message}],
        })

        return messages
