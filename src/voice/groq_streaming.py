"""
Groq Streaming LLM Module for CropFresh Voice Agent
=====================================================

Provides real-time streaming chat completions via Groq LPU.
Designed for ultra-low latency voice agents with:
- Token-by-token streaming
- Sentence boundary detection for speculative TTS
- Cancellation support for barge-in interruption
- System prompt injection for CropFresh agent persona

Usage:
    streamer = GroqLLMStreaming()
    async for sentence in streamer.stream_sentences("What is the price of wheat?"):
        # Send each sentence to TTS immediately
        await tts.synthesize_stream(sentence)
"""

import os
import re
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from loguru import logger

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512

# Sentence boundary regex — splits on ., !, ?, |, ।
# Includes Hindi danda (।) for Indic language support.
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?।\|])\s+")

# CropFresh system prompt for the voice agent persona
CROPFRESH_SYSTEM_PROMPT = """You are CropFresh AI Voice Assistant — a helpful agricultural marketplace assistant for Indian farmers.

Rules:
- Keep responses SHORT (1-3 sentences max) since they will be spoken aloud.
- Use simple language. Farmers may speak Hindi, Kannada, Telugu, Tamil, or English.
- Match the user's language carefully. Remember context and farmer details across the conversation.
- If the user explicitly asks to switch languages (e.g. "switch to Kannada", "speak in English"), you MUST start your response with the exact tag `[LANG:code]` where code is en, kn, te, ta, or hi.
  Example: `[LANG:kn] ಖಂಡಿತ, ನಾನು ಈಗ ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡುತ್ತೇನೆ.`
- Be warm, respectful, and practical.
- Help with: crop prices, listing produce, finding buyers, weather, farming advice.
- Never use markdown formatting, bullet points, or special characters — plain spoken text only."""


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class StreamingConfig:
    """Configuration for streaming LLM."""
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    system_prompt: str = CROPFRESH_SYSTEM_PROMPT


@dataclass
class SentenceChunk:
    """A sentence-boundary chunk from the LLM stream."""
    text: str
    is_final: bool = False
    token_count: int = 0


# ═══════════════════════════════════════════════════════════════
# Groq Streaming LLM
# ═══════════════════════════════════════════════════════════════

class GroqLLMStreaming:
    """
    Streaming LLM wrapper for Groq API.

    Streams tokens from Groq and yields complete sentences
    as soon as a sentence boundary is detected. This enables
    speculative TTS — the TTS engine can start speaking the
    first sentence while the LLM is still generating the rest.

    Usage:
        streamer = GroqLLMStreaming()

        # Stream individual tokens
        async for token in streamer.stream_tokens("Hello"):
            print(token, end="")

        # Stream complete sentences (for TTS pipeline)
        async for chunk in streamer.stream_sentences("Hello"):
            await tts.synthesize(chunk.text)
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._config = config or StreamingConfig()
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._client = None
        self._cancelled = False

        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Provide via constructor or .env."
            )

    def _ensure_client(self) -> None:
        """Lazy-init the Groq async client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self._api_key)
                logger.info("[GroqStreaming] Client initialized")
            except ImportError:
                raise ImportError(
                    "groq package not installed. Run: uv pip install groq"
                )

    def cancel(self) -> None:
        """Cancel the current stream (for barge-in)."""
        self._cancelled = True
        logger.info("[GroqStreaming] Stream cancelled (barge-in)")

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
        Stream individual tokens from Groq.

        Args:
            user_message: The user's text input.
            conversation_history: Optional list of prior messages.
            language: Language code for system prompt adaptation.

        Yields:
            Individual tokens as strings.
        """
        self._ensure_client()
        self.reset()

        messages = self._build_messages(
            user_message, conversation_history, language
        )

        try:
            stream = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if self._cancelled:
                    logger.info("[GroqStreaming] Token stream cancelled")
                    break

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            logger.error(f"[GroqStreaming] Stream error: {e}")
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
        Stream complete sentences from Groq.

        Buffers tokens until a sentence boundary is detected,
        then yields the complete sentence. This enables the
        TTS engine to start speaking while the LLM continues.

        Args:
            user_message: The user's text input.
            conversation_history: Optional prior messages.
            language: Language code.

        Yields:
            SentenceChunk with complete sentence text.
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
                # We have at least one complete sentence
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        yield SentenceChunk(
                            text=sentence,
                            is_final=False,
                            token_count=token_count,
                        )
                # Keep the remaining partial sentence in buffer
                buffer = sentences[-1]

        # Flush any remaining text as the final chunk
        if buffer.strip() and not self._cancelled:
            yield SentenceChunk(
                text=buffer.strip(),
                is_final=True,
                token_count=token_count,
            )

    # ──────────────────────────────────────────────────────────
    # Non-streaming (for simple use cases)
    # ──────────────────────────────────────────────────────────

    async def generate(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        language: str = "en",
    ) -> str:
        """
        Generate a complete response (non-streaming).

        Args:
            user_message: The user's text input.

        Returns:
            Complete response text.
        """
        tokens = []
        async for token in self.stream_tokens(
            user_message, conversation_history, language
        ):
            tokens.append(token)
        return "".join(tokens)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _build_messages(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]],
        language: str,
    ) -> list[dict]:
        """Build the messages array for the Groq API call."""
        # Adapt system prompt for language
        system_prompt = self._config.system_prompt
        if language != "en":
            lang_names = {
                "hi": "Hindi", "kn": "Kannada", "te": "Telugu",
                "ta": "Tamil", "ml": "Malayalam", "mr": "Marathi",
                "gu": "Gujarati", "bn": "Bengali", "pa": "Punjabi",
                "or": "Odia",
            }
            lang_name = lang_names.get(language, language)
            system_prompt += (
                f"\n\nIMPORTANT: The user is speaking in {lang_name}. "
                f"You MUST respond in {lang_name}."
            )

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (limit to last 20 messages for robust memory)
        if conversation_history:
            messages.extend(conversation_history[-20:])

        messages.append({"role": "user", "content": user_message})

        return messages
