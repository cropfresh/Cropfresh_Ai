"""
Multi-provider STT with automatic fallback.

Priority order:
1. Faster Whisper (fastest, local, CPU-friendly)
2. IndicConformer (best for Indian languages, needs GPU)
3. Groq Whisper API (cloud fallback)
"""

from loguru import logger

from src.voice.stt_pkg.faster_whisper import FasterWhisperSTT
from src.voice.stt_pkg.groq_whisper import GroqWhisperSTT
from src.voice.stt_pkg.indic_whisper import IndicWhisperSTT
from src.voice.stt_pkg.models import SupportedLanguage, TranscriptionResult


class MultiProviderSTT:
    """Multi-provider Speech-to-Text with automatic fallback."""

    def __init__(
        self,
        use_faster_whisper: bool = True,
        use_indicconformer: bool = False,
        faster_whisper_model: str = "small",
    ):
        self._providers = []
        self._provider_names = []

        if use_faster_whisper:
            try:
                self._providers.append(
                    FasterWhisperSTT(model_size=faster_whisper_model),
                )
                self._provider_names.append("faster-whisper")
                logger.info("MultiProviderSTT: Faster Whisper enabled")
            except Exception as e:
                logger.warning(f"Could not init Faster Whisper: {e}")

        if use_indicconformer:
            try:
                self._providers.append(IndicWhisperSTT())
                self._provider_names.append("indicconformer")
                logger.info("MultiProviderSTT: AI4Bharat IndicConformer enabled")
            except Exception as e:
                logger.warning(f"Could not init IndicConformer: {e}")

        try:
            self._providers.append(GroqWhisperSTT())
            self._provider_names.append("groq_whisper")
            logger.info("MultiProviderSTT: GroqWhisperSTT registered as cloud fallback")
        except ValueError:
            pass
        except Exception as e:
            logger.warning(f"GroqWhisperSTT unavailable: {e}")

        if not self._providers:
            raise RuntimeError(
                "No STT providers available. Set GROQ_API_KEY for cloud fallback, "
                "or install faster-whisper for local inference."
            )

        logger.info(
            f"MultiProviderSTT initialized with {len(self._providers)} "
            f"providers: {self._provider_names}"
        )

    async def transcribe(
        self,
        audio: bytes,
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe audio using the best available provider (fallback chain)."""
        errors = []
        for i, provider in enumerate(self._providers):
            try:
                logger.debug(f"Trying STT provider: {self._provider_names[i]}")
                result = await provider.transcribe(audio, language)

                if result.is_successful:
                    logger.info(
                        f"STT success with {self._provider_names[i]}: "
                        f"{result.text[:50]}..."
                    )
                    return result
                else:
                    logger.warning(
                        f"STT provider {self._provider_names[i]} returned "
                        f"empty/low-confidence result. text='{result.text}'"
                    )
                    errors.append(f"{self._provider_names[i]}: empty result")

            except Exception as e:
                logger.warning(f"STT provider {self._provider_names[i]} failed: {e}")
                errors.append(f"{self._provider_names[i]}: {str(e)}")
                continue

        error_msg = "; ".join(errors)
        logger.error(f"All local STT providers failed: {error_msg}")

        return TranscriptionResult(
            text="",
            language=language if language != "auto" else "hi",
            confidence=0.0,
            duration_seconds=0.0,
            provider="none",
        )

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names"""
        return self._provider_names.copy()

    def get_supported_languages(self) -> list[str]:
        """Get supported language codes from first available provider."""
        for provider in self._providers:
            if hasattr(provider, "get_supported_languages"):
                return provider.get_supported_languages()
        return [lang.value for lang in SupportedLanguage if lang != SupportedLanguage.AUTO]
