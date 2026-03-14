"""
Silero VAD Core
===============
Silero Voice Activity Detection integration.
"""

from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from .models import SpeechSegment, VADEvent, VADState


class SileroVAD:
    """
    Silero VAD for real-time voice activity detection.

    Uses the Silero VAD v5 ONNX model for efficient inference.
    Designed for real-time streaming with low latency.
    """

    # Model configuration
    SAMPLE_RATE = 16000
    CHUNK_SIZE_MS = 30  # Silero works best with 30ms chunks
    CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SIZE_MS // 1000  # 480 samples

    # Thresholds
    DEFAULT_THRESHOLD = 0.5  # Speech probability threshold
    SPEECH_PAD_MS = 300  # Padding before/after speech
    MIN_SPEECH_DURATION_MS = 250  # Minimum speech duration
    MIN_SILENCE_DURATION_MS = 300  # Silence before considering speech end

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        sample_rate: int = SAMPLE_RATE,
        min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS,
        speech_pad_ms: int = SPEECH_PAD_MS,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        self._model = None
        self._session = None
        self._state = None
        self._h = None
        self._c = None
        self._use_v5_format = False
        self._initialized = False

        # State tracking
        self._is_speaking = False
        self._speech_start_ms = 0.0
        self._silence_start_ms = 0.0
        self._current_segment_audio: list[bytes] = []
        self._timestamp_ms = 0.0

        # Chunk size for this sample rate
        self.chunk_samples = sample_rate * self.CHUNK_SIZE_MS // 1000

        logger.info(f"SileroVAD initialized: threshold={threshold}, sr={sample_rate}")

    async def initialize(self) -> None:
        """Load the Silero VAD model"""
        if self._initialized:
            return

        try:
            import onnxruntime as ort

            model_path = await self._get_model_path()

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 1

            self._session = ort.InferenceSession(
                model_path,
                session_options,
                providers=['CPUExecutionProvider']
            )

            input_names = [inp.name for inp in self._session.get_inputs()]
            logger.debug(f"VAD model inputs: {input_names}")

            if 'state' in input_names:
                self._use_v5_format = True
                logger.info("Detected Silero VAD v5 format (state input)")
            else:
                self._use_v5_format = False
                logger.info("Detected Silero VAD v4 format (h/c inputs)")

            self._reset_states()

            self._initialized = True
            logger.info("Silero VAD model loaded successfully")

        except ImportError:
            logger.error("onnxruntime not installed. Run: pip install onnxruntime")
            raise
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    async def _get_model_path(self) -> str:
        """Get path to Silero VAD ONNX model"""
        local_paths = [
            Path(__file__).parent / "models" / "silero_vad.onnx",
            Path.home() / ".cache" / "silero_vad" / "silero_vad.onnx",
        ]

        for path in local_paths:
            if path.exists():
                return str(path)

        logger.info("Downloading Silero VAD model...")

        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
                onnx=True,
            )

            if hasattr(model, 'model_path'):
                return model.model_path

            onnx_path = Path.home() / ".cache" / "silero_vad" / "silero_vad.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)

            import urllib.request
            url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            urllib.request.urlretrieve(url, str(onnx_path))

            return str(onnx_path)

        except Exception as e:
            logger.warning(f"Could not download via torch.hub: {e}")

            onnx_path = Path.home() / ".cache" / "silero_vad" / "silero_vad.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)

            import urllib.request
            url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            urllib.request.urlretrieve(url, str(onnx_path))

            return str(onnx_path)

    def _reset_states(self) -> None:
        if self._use_v5_format:
            state_dim = 128 if self.sample_rate == 16000 else 64
            self._state = np.zeros((2, 1, state_dim), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

        self._is_speaking = False
        self._speech_start_ms = 0.0
        self._silence_start_ms = 0.0
        self._current_segment_audio = []
        self._timestamp_ms = 0.0

    def reset(self) -> None:
        self._reset_states()
        logger.debug("VAD state reset")

    def _audio_to_float(self, audio_bytes: bytes) -> np.ndarray:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        return audio_float

    def process_chunk(self, audio_chunk: bytes) -> VADEvent:
        if not self._initialized:
            raise RuntimeError("VAD not initialized. Call initialize() first.")

        audio = self._audio_to_float(audio_chunk)

        if len(audio) != self.chunk_samples:
            if len(audio) < self.chunk_samples:
                audio = np.pad(audio, (0, self.chunk_samples - len(audio)))
            else:
                audio = audio[:self.chunk_samples]

        if self._use_v5_format:
            ort_inputs = {
                'input': audio.reshape(1, -1),
                'state': self._state,
                'sr': np.array([self.sample_rate], dtype=np.int64),
            }
            output, self._state = self._session.run(None, ort_inputs)
        else:
            ort_inputs = {
                'input': audio.reshape(1, -1),
                'h': self._h,
                'c': self._c,
                'sr': np.array([self.sample_rate], dtype=np.int64),
            }
            output, self._h, self._c = self._session.run(None, ort_inputs)

        probability = float(output[0][0])
        chunk_duration_ms = len(audio) * 1000 / self.sample_rate
        self._timestamp_ms += chunk_duration_ms

        is_speech = probability >= self.threshold

        if is_speech:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_ms = self._timestamp_ms
                self._current_segment_audio = [audio_chunk]
                state = VADState.SPEECH_START
            else:
                self._current_segment_audio.append(audio_chunk)
                state = VADState.SPEECH
        else:
            if self._is_speaking:
                if self._silence_start_ms == 0:
                    self._silence_start_ms = self._timestamp_ms

                silence_duration = self._timestamp_ms - self._silence_start_ms

                if silence_duration >= self.min_silence_duration_ms:
                    self._is_speaking = False
                    self._silence_start_ms = 0
                    state = VADState.SPEECH_END
                else:
                    self._current_segment_audio.append(audio_chunk)
                    state = VADState.SPEECH
            else:
                state = VADState.SILENCE

        return VADEvent(
            state=state,
            timestamp_ms=self._timestamp_ms,
            probability=probability,
            audio_chunk=audio_chunk if is_speech else None,
        )

    async def process_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[VADEvent]:
        if not self._initialized:
            await self.initialize()

        async for chunk in audio_stream:
            event = self.process_chunk(chunk)
            yield event

    def get_speech_segment(self) -> Optional[SpeechSegment]:
        if not self._current_segment_audio:
            return None

        audio_bytes = b"".join(self._current_segment_audio)

        segment = SpeechSegment(
            audio=audio_bytes,
            start_ms=self._speech_start_ms,
            end_ms=self._timestamp_ms,
            duration_ms=self._timestamp_ms - self._speech_start_ms,
            sample_rate=self.sample_rate,
        )

        self._current_segment_audio = []
        return segment
