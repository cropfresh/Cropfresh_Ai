"""
Duplex Pipeline Initializers
============================
Mixin handling the initialization of STT, TTS, and LLM components.
"""

from loguru import logger


class InitializersMixin:
    """Mixin for initializing pipeline components (LLM, STT, TTS)."""

    async def _init_llm(self) -> None:
        """Initialize the LLM streaming provider."""
        try:
            if self._llm_provider == "groq":
                from src.voice.groq_streaming import GroqLLMStreaming
                self._llm = GroqLLMStreaming()
                logger.info("[DuplexPipeline] LLM: Groq initialized")
            elif self._llm_provider == "bedrock":
                from src.voice.bedrock_streaming import BedrockLLMStreaming
                self._llm = BedrockLLMStreaming()
                logger.info("[DuplexPipeline] LLM: Bedrock initialized")
            else:
                raise ValueError(f"Unknown LLM provider: {self._llm_provider}")
        except Exception as e:
            logger.warning(f"[DuplexPipeline] Primary LLM ({self._llm_provider}) failed: {e}")
            # Fallback
            if self._llm_provider == "groq":
                try:
                    from src.voice.bedrock_streaming import BedrockLLMStreaming
                    self._llm = BedrockLLMStreaming()
                    self._llm_provider = "bedrock"
                    logger.info("[DuplexPipeline] LLM: Fell back to Bedrock")
                except Exception as e2:
                    logger.error(f"[DuplexPipeline] Bedrock fallback also failed: {e2}")
                    raise

    async def _init_stt(self) -> None:
        """Initialize the STT provider."""
        try:
            from src.voice.stt import MultiProviderSTT
            self._stt = MultiProviderSTT(
                use_faster_whisper=True,
                use_indicconformer=False,
            )
            logger.info("[DuplexPipeline] STT: MultiProvider initialized")
        except Exception as e:
            logger.error(f"[DuplexPipeline] STT init failed: {e}")
            raise

    async def _init_tts(self) -> None:
        """Initialize the TTS streaming provider."""
        try:
            from src.voice.streaming_tts import StreamingTTS, StreamingTTSProvider
            provider = StreamingTTSProvider.EDGE
            self._tts = StreamingTTS(preferred_provider=provider)
            logger.info("[DuplexPipeline] TTS: Edge TTS initialized")
        except Exception as e:
            logger.error(f"[DuplexPipeline] TTS init failed: {e}")
            raise
