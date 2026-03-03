# Voice Module for CropFresh AI
# STT: Multi-provider (Faster Whisper, IndicConformer, Groq)
# TTS: Edge TTS (primary), Streaming TTS with barge-in support
# VAD: Silero VAD for real-time voice activity detection
# Transport: WebRTC for bidirectional streaming

from .stt import IndicWhisperSTT, FasterWhisperSTT, MultiProviderSTT, TranscriptionResult
from .tts import IndicTTS, EdgeTTSProvider, SynthesisResult
from .entity_extractor import VoiceEntityExtractor, ExtractionResult
from .audio_utils import AudioProcessor

# New modules for advanced voice agent
try:
    from .vad import SileroVAD, VADState, VADEvent, SpeechSegment, BargeinDetector
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    from .webrtc_transport import WebRTCTransport, WebRTCSignaling, ConnectionState
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    from .streaming_tts import StreamingTTS, CancellationToken, AudioChunk, MultiProviderStreamingTTS
    STREAMING_TTS_AVAILABLE = True
except ImportError:
    STREAMING_TTS_AVAILABLE = False

try:
    from .duplex_pipeline import DuplexPipeline, PipelineState, PipelineEvent, AudioOutputChunk
    DUPLEX_AVAILABLE = True
except ImportError:
    DUPLEX_AVAILABLE = False

try:
    from .groq_streaming import GroqLLMStreaming, SentenceChunk
    GROQ_STREAMING_AVAILABLE = True
except ImportError:
    GROQ_STREAMING_AVAILABLE = False

__all__ = [
    # STT
    "IndicWhisperSTT",
    "FasterWhisperSTT",
    "MultiProviderSTT",
    "TranscriptionResult",
    # TTS
    "IndicTTS",
    "EdgeTTSProvider",
    "SynthesisResult",
    # Entity Extraction
    "VoiceEntityExtractor",
    "ExtractionResult",
    # Audio Processing
    "AudioProcessor",
    # VAD (if available)
    "SileroVAD",
    "VADState",
    "VADEvent",
    "SpeechSegment",
    "BargeinDetector",
    "VAD_AVAILABLE",
    # WebRTC (if available)
    "WebRTCTransport",
    "WebRTCSignaling",
    "ConnectionState",
    "WEBRTC_AVAILABLE",
    # Streaming TTS (if available)
    "StreamingTTS",
    "CancellationToken",
    "AudioChunk",
    "MultiProviderStreamingTTS",
    "STREAMING_TTS_AVAILABLE",
    # Duplex Pipeline (if available)
    "DuplexPipeline",
    "PipelineState",
    "PipelineEvent",
    "AudioOutputChunk",
    "DUPLEX_AVAILABLE",
    # Groq Streaming (if available)
    "GroqLLMStreaming",
    "SentenceChunk",
    "GROQ_STREAMING_AVAILABLE",
]

