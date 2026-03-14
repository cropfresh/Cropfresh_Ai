"""
Barge-in Detector
=================
Uses VAD to detect user interruptions.
"""

from typing import Callable, Optional

from loguru import logger

from .models import VADEvent
from .silero import SileroVAD


class BargeinDetector:
    """
    Barge-in detection for interrupting AI responses.

    Monitors audio input during TTS playback and triggers
    interruption when user starts speaking.
    """

    BARGEIN_THRESHOLD = 0.6  # Higher threshold for barge-in
    BARGEIN_DURATION_MS = 150  # Speech duration to trigger barge-in

    def __init__(
        self,
        vad: SileroVAD,
        threshold: float = BARGEIN_THRESHOLD,
        duration_ms: int = BARGEIN_DURATION_MS,
    ):
        self.vad = vad
        self.threshold = threshold
        self.duration_ms = duration_ms

        self._is_monitoring = False
        self._speech_detected_ms = 0.0
        self._on_bargein: Optional[Callable[[], None]] = None

    @property
    def on_bargein(self) -> Optional[Callable[[], None]]:
        return self._on_bargein

    @on_bargein.setter
    def on_bargein(self, callback: Callable[[], None]) -> None:
        self._on_bargein = callback

    def start_monitoring(self) -> None:
        self._is_monitoring = True
        self._speech_detected_ms = 0.0
        logger.debug("Barge-in monitoring started")

    def stop_monitoring(self) -> None:
        self._is_monitoring = False
        self._speech_detected_ms = 0.0
        logger.debug("Barge-in monitoring stopped")

    def check_bargein(self, event: VADEvent) -> bool:
        if not self._is_monitoring:
            return False

        if event.probability >= self.threshold:
            if self._speech_detected_ms == 0:
                self._speech_detected_ms = event.timestamp_ms

            speech_duration = event.timestamp_ms - self._speech_detected_ms

            if speech_duration >= self.duration_ms:
                logger.info(f"Barge-in triggered after {speech_duration:.0f}ms of speech")

                if self._on_bargein:
                    self._on_bargein()

                self.stop_monitoring()
                return True
        else:
            self._speech_detected_ms = 0.0

        return False
