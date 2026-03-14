"""
CropFresh Agent Processor for Pipecat
======================================

A Pipecat FrameProcessor that bridges transcribed text (from
LocalBhashiniSTTService) → VoiceAgent intent router → response text
(fed into LocalBhashiniTTSService).

This replaces the raw OpenAILLMService used in the prototype pipeline
with CropFresh's fully-featured VoiceAgent (multi-turn, multi-language,
intent routing for all 12 supported intents).

Frame contract
--------------
Receives:  TextFrame  (transcription from STT)
Pushes:    TextFrame  (agent response text for TTS)
All other frames are passed through unchanged.
"""

from loguru import logger
from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class CropFreshAgentProcessor(FrameProcessor):
    """
    Pipecat FrameProcessor that routes transcribed text through VoiceAgent.

    Replaces OpenAILLMService in the CropFresh Pipecat pipeline.

    VoiceAgent handles:
    - Multi-turn session management (CREATE_LISTING, FIND_BUYER, REGISTER)
    - Intent extraction + routing (12 intents)
    - Multi-language template responses (en / hi / kn)
    - Optional LLM escalation for advisory queries

    Args:
        voice_agent:  Fully-initialised VoiceAgent instance.
        session_id:   Pipecat session identifier (maps to VoiceSession).
        language:     ISO language code – 'hi', 'kn', or 'en'. Default 'hi'.
    """

    # Fallback messages for when VoiceAgent raises an unexpected exception.
    _FALLBACK = {
        "hi": "माफ करें, कुछ समस्या आई। कृपया फिर से बोलें।",
        "kn": "ಕ್ಷಮಿಸಿ, ಸಮಸ್ಯೆ ಬಂದಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ.",
        "en": "Sorry, something went wrong. Please try again.",
    }

    def __init__(
        self,
        voice_agent,
        session_id: str,
        language: str = "hi",
    ) -> None:
        super().__init__()
        self._voice_agent = voice_agent
        self._session_id = session_id
        self._user_id = session_id          # use session_id as proxy user_id
        self._language = language

        logger.info(
            f"[AgentProcessor] Initialised — session={session_id!r}, lang={language!r}"
        )

    # ------------------------------------------------------------------
    # Public helpers (useful for testing without a live pipeline)
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def language(self) -> str:
        return self._language

    # ------------------------------------------------------------------
    # FrameProcessor core
    # ------------------------------------------------------------------

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        """
        Handle incoming frames.

        - TextFrame with non-empty text → route through VoiceAgent.
        - TextFrame that is blank/whitespace → pass through silently.
        - All other frame types → pass through unchanged.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            text = frame.text.strip() if frame.text else ""
            if text:
                response_text = await self._route_text(text)
                await self.push_frame(TextFrame(text=response_text))
            else:
                # Empty frame — pass through so downstream processors
                # (e.g., TTS silence handling) can decide what to do.
                await self.push_frame(frame, direction)
        else:
            # AudioRawFrame, EndFrame, LLMMessagesFrame, etc. → pass through.
            await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _route_text(self, text: str) -> str:
        """
        Call VoiceAgent.handle_text_input and return the response string.

        Catches all exceptions and returns a language-appropriate fallback
        so the TTS stage always receives something to speak.
        """
        logger.info(f"[AgentProcessor] Routing text → VoiceAgent: {text!r}")
        try:
            response = await self._voice_agent.handle_text_input(
                text=text,
                user_id=self._user_id,
                session_id=self._session_id,
                language=self._language,
            )
            reply = response.response_text
            logger.info(f"[AgentProcessor] VoiceAgent reply: {reply!r}")
            return reply
        except Exception as exc:
            logger.error(f"[AgentProcessor] VoiceAgent raised: {exc}")
            return self._FALLBACK.get(self._language, self._FALLBACK["en"])
