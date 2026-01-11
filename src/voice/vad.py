"""
Voice Activity Detection (VAD) Module for CropFresh Voice Agent

Uses Silero VAD for high-accuracy, real-time voice activity detection.
Enables: barge-in support, turn-taking, and efficient audio processing.

Features:
- 87.7% TPR @ 5% FPR (much better than WebRTC VAD's 50%)
- 1.8MB model size, processes 30ms chunks in ~1ms
- Supports 8kHz and 16kHz audio
- Works offline with ONNX runtime
"""

import asyncio
import io
import struct
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Callable, Optional

import numpy as np
from loguru import logger


class VADState(Enum):
    """Voice activity detection states"""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH = "speech"
    SPEECH_END = "speech_end"


@dataclass
class VADEvent:
    """Event from VAD processing"""
    state: VADState
    timestamp_ms: float
    probability: float
    audio_chunk: Optional[bytes] = None
    

@dataclass
class SpeechSegment:
    """Complete speech segment with audio data"""
    audio: bytes
    start_ms: float
    end_ms: float
    duration_ms: float
    sample_rate: int = 16000


class SileroVAD:
    """
    Silero VAD for real-time voice activity detection.
    
    Uses the Silero VAD v5 ONNX model for efficient inference.
    Designed for real-time streaming with low latency.
    
    Usage:
        vad = SileroVAD()
        await vad.initialize()
        
        # Process audio chunks
        async for event in vad.process_stream(audio_stream):
            if event.state == VADState.SPEECH_START:
                print("User started speaking")
            elif event.state == VADState.SPEECH_END:
                print("User stopped speaking")
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
        """
        Initialize Silero VAD.
        
        Args:
            threshold: Speech probability threshold (0.0 - 1.0)
            sample_rate: Audio sample rate (8000 or 16000)
            min_speech_duration_ms: Minimum speech duration to trigger
            min_silence_duration_ms: Silence duration before speech end
            speech_pad_ms: Padding around speech segments
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        self._model = None
        self._session = None
        self._state = None  # Combined state for v5
        self._h = None  # Hidden state for older versions
        self._c = None  # Cell state for older versions
        self._use_v5_format = False  # Flag for model version
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
            import os
            
            # First try to load from local cache/huggingface
            model_path = await self._get_model_path()
            
            # Create ONNX inference session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 1
            
            self._session = ort.InferenceSession(
                model_path,
                session_options,
                providers=['CPUExecutionProvider']
            )
            
            # Detect model version by checking input names
            input_names = [inp.name for inp in self._session.get_inputs()]
            logger.debug(f"VAD model inputs: {input_names}")
            
            if 'state' in input_names:
                self._use_v5_format = True
                logger.info("Detected Silero VAD v5 format (state input)")
            else:
                self._use_v5_format = False
                logger.info("Detected Silero VAD v4 format (h/c inputs)")
            
            # Initialize hidden states
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
        import os
        from pathlib import Path
        
        # Check for local model first
        local_paths = [
            Path(__file__).parent / "models" / "silero_vad.onnx",
            Path.home() / ".cache" / "silero_vad" / "silero_vad.onnx",
        ]
        
        for path in local_paths:
            if path.exists():
                return str(path)
        
        # Download from torch.hub (will cache locally)
        logger.info("Downloading Silero VAD model...")
        
        try:
            import torch
            
            # Use torch.hub to get the model and save as ONNX
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
                onnx=True,  # Get ONNX version
            )
            
            # The ONNX model path is returned
            if hasattr(model, 'model_path'):
                return model.model_path
            
            # Alternative: download directly
            onnx_path = Path.home() / ".cache" / "silero_vad" / "silero_vad.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            
            import urllib.request
            url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            urllib.request.urlretrieve(url, str(onnx_path))
            
            return str(onnx_path)
            
        except Exception as e:
            logger.warning(f"Could not download via torch.hub: {e}")
            
            # Direct download fallback
            onnx_path = Path.home() / ".cache" / "silero_vad" / "silero_vad.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            
            import urllib.request
            url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            urllib.request.urlretrieve(url, str(onnx_path))
            
            return str(onnx_path)
    
    def _reset_states(self) -> None:
        """Reset LSTM hidden states"""
        if self._use_v5_format:
            # Silero VAD v5 uses a single state tensor
            # Shape: (2, 1, 128) for 16kHz or (2, 1, 64) for 8kHz
            state_dim = 128 if self.sample_rate == 16000 else 64
            self._state = np.zeros((2, 1, state_dim), dtype=np.float32)
        else:
            # Silero VAD v4 uses separate h and c tensors
            # Shape: (2, 1, 64)
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        
        # Reset state tracking
        self._is_speaking = False
        self._speech_start_ms = 0.0
        self._silence_start_ms = 0.0
        self._current_segment_audio = []
        self._timestamp_ms = 0.0
    
    def reset(self) -> None:
        """Reset VAD state for new session"""
        self._reset_states()
        logger.debug("VAD state reset")
    
    def _audio_to_float(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to float32 numpy array"""
        # Assume 16-bit PCM audio
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        return audio_float
    
    def process_chunk(self, audio_chunk: bytes) -> VADEvent:
        """
        Process a single audio chunk and return VAD event.
        
        Args:
            audio_chunk: Audio bytes (16-bit PCM, mono)
            
        Returns:
            VADEvent with current state and probability
        """
        if not self._initialized:
            raise RuntimeError("VAD not initialized. Call initialize() first.")
        
        # Convert to float
        audio = self._audio_to_float(audio_chunk)
        
        # Ensure correct chunk size
        if len(audio) != self.chunk_samples:
            # Pad or truncate
            if len(audio) < self.chunk_samples:
                audio = np.pad(audio, (0, self.chunk_samples - len(audio)))
            else:
                audio = audio[:self.chunk_samples]
        
        # Run inference with correct input format based on model version
        if self._use_v5_format:
            # Silero VAD v5 format
            ort_inputs = {
                'input': audio.reshape(1, -1),
                'state': self._state,
                'sr': np.array([self.sample_rate], dtype=np.int64),
            }
            output, self._state = self._session.run(None, ort_inputs)
        else:
            # Silero VAD v4 format
            ort_inputs = {
                'input': audio.reshape(1, -1),
                'h': self._h,
                'c': self._c,
                'sr': np.array([self.sample_rate], dtype=np.int64),
            }
            output, self._h, self._c = self._session.run(None, ort_inputs)
        
        probability = float(output[0][0])
        
        # Update timestamp
        chunk_duration_ms = len(audio) * 1000 / self.sample_rate
        self._timestamp_ms += chunk_duration_ms
        
        # Determine state
        is_speech = probability >= self.threshold
        
        if is_speech:
            if not self._is_speaking:
                # Speech started
                self._is_speaking = True
                self._speech_start_ms = self._timestamp_ms
                self._current_segment_audio = [audio_chunk]
                state = VADState.SPEECH_START
            else:
                # Continuing speech
                self._current_segment_audio.append(audio_chunk)
                state = VADState.SPEECH
        else:
            if self._is_speaking:
                # Possible speech end
                if self._silence_start_ms == 0:
                    self._silence_start_ms = self._timestamp_ms
                
                silence_duration = self._timestamp_ms - self._silence_start_ms
                
                if silence_duration >= self.min_silence_duration_ms:
                    # Speech ended
                    self._is_speaking = False
                    self._silence_start_ms = 0
                    state = VADState.SPEECH_END
                else:
                    # Still within speech (short silence)
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
        """
        Process an async audio stream and yield VAD events.
        
        Args:
            audio_stream: Async iterator yielding audio chunks
            
        Yields:
            VADEvent for each processed chunk
        """
        if not self._initialized:
            await self.initialize()
        
        async for chunk in audio_stream:
            event = self.process_chunk(chunk)
            yield event
    
    def get_speech_segment(self) -> Optional[SpeechSegment]:
        """
        Get the current accumulated speech segment.
        
        Returns:
            SpeechSegment if speech was collected, None otherwise
        """
        if not self._current_segment_audio:
            return None
        
        # Combine all audio chunks
        audio_bytes = b"".join(self._current_segment_audio)
        
        segment = SpeechSegment(
            audio=audio_bytes,
            start_ms=self._speech_start_ms,
            end_ms=self._timestamp_ms,
            duration_ms=self._timestamp_ms - self._speech_start_ms,
            sample_rate=self.sample_rate,
        )
        
        # Clear buffer
        self._current_segment_audio = []
        
        return segment


class BargeinDetector:
    """
    Barge-in detection for interrupting AI responses.
    
    Monitors audio input during TTS playback and triggers
    interruption when user starts speaking.
    
    Usage:
        detector = BargeinDetector(vad)
        detector.on_bargein = lambda: stop_tts_playback()
        detector.start_monitoring()
    """
    
    # Barge-in configuration
    BARGEIN_THRESHOLD = 0.6  # Higher threshold for barge-in
    BARGEIN_DURATION_MS = 150  # Speech duration to trigger barge-in
    
    def __init__(
        self,
        vad: SileroVAD,
        threshold: float = BARGEIN_THRESHOLD,
        duration_ms: int = BARGEIN_DURATION_MS,
    ):
        """
        Initialize barge-in detector.
        
        Args:
            vad: SileroVAD instance
            threshold: Probability threshold for barge-in
            duration_ms: Speech duration to trigger
        """
        self.vad = vad
        self.threshold = threshold
        self.duration_ms = duration_ms
        
        self._is_monitoring = False
        self._speech_detected_ms = 0.0
        self._on_bargein: Optional[Callable[[], None]] = None
        
    @property
    def on_bargein(self) -> Optional[Callable[[], None]]:
        """Callback when barge-in is detected"""
        return self._on_bargein
    
    @on_bargein.setter
    def on_bargein(self, callback: Callable[[], None]) -> None:
        self._on_bargein = callback
    
    def start_monitoring(self) -> None:
        """Start monitoring for barge-in"""
        self._is_monitoring = True
        self._speech_detected_ms = 0.0
        logger.debug("Barge-in monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring for barge-in"""
        self._is_monitoring = False
        self._speech_detected_ms = 0.0
        logger.debug("Barge-in monitoring stopped")
    
    def check_bargein(self, event: VADEvent) -> bool:
        """
        Check if barge-in should be triggered.
        
        Args:
            event: VAD event to check
            
        Returns:
            True if barge-in triggered
        """
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
            # Reset if silence detected
            self._speech_detected_ms = 0.0
        
        return False


# Utility functions

def create_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Create silent audio bytes"""
    num_samples = int(sample_rate * duration_ms / 1000)
    silence = np.zeros(num_samples, dtype=np.int16)
    return silence.tobytes()


def bytes_to_wav(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Convert raw PCM bytes to WAV format"""
    num_channels = 1
    sample_width = 2  # 16-bit
    byte_rate = sample_rate * num_channels * sample_width
    block_align = num_channels * sample_width
    data_size = len(audio_bytes)
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,  # PCM format size
        1,   # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        16,  # bits per sample
        b'data',
        data_size,
    )
    
    return header + audio_bytes
