"""
WebRTC Transport Tracks
=======================
Custom media stream tracks for sending and receiving audio over WebRTC.
"""

import asyncio
from typing import AsyncIterator

import numpy as np
from loguru import logger

from .models import AudioChunk

try:
    from aiortc.mediastreams import MediaStreamTrack
    from av import AudioFrame
    AIORTC_AVAILABLE = True
except ImportError:
    MediaStreamTrack = object
    AudioFrame = object
    AIORTC_AVAILABLE = False
    logger.warning("aiortc not installed. WebRTC disabled.")


class AudioReceiveTrack(MediaStreamTrack if AIORTC_AVAILABLE else object):
    """
    Custom audio track for receiving audio from WebRTC peer.
    Collects audio frames and exposes them via async iterator.
    """
    
    kind = "audio"
    
    def __init__(self, sample_rate: int = 16000):
        if AIORTC_AVAILABLE:
            super().__init__()
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=100)
        self._running = True
        self._timestamp_ms = 0.0
    
    async def recv(self) -> "AudioFrame":
        """Receive audio frame (called by aiortc)"""
        frame = AudioFrame(format="s16", layout="mono", samples=480)
        frame.sample_rate = self.sample_rate
        return frame
    
    def on_frame(self, frame: "AudioFrame") -> None:
        """Handle incoming audio frame"""
        try:
            audio = frame.to_ndarray()
            if len(audio.shape) > 1:
                audio = audio.mean(axis=0)  # Convert to mono
            
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            chunk = AudioChunk(
                data=audio_bytes,
                sample_rate=frame.sample_rate,
                channels=1,
                timestamp_ms=self._timestamp_ms,
                samples=len(audio_int16),
            )
            
            self._timestamp_ms += len(audio_int16) * 1000 / frame.sample_rate
            
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(chunk)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
    
    async def get_audio(self) -> AsyncIterator[AudioChunk]:
        """Get audio chunks as async iterator"""
        while self._running:
            try:
                chunk = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                yield chunk
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error getting audio: {e}")
                break
    
    def stop(self) -> None:
        """Stop receiving audio"""
        self._running = False


class AudioSendTrack(MediaStreamTrack if AIORTC_AVAILABLE else object):
    """
    Custom audio track for sending audio to WebRTC peer.
    Accepts audio bytes and converts to WebRTC frames.
    """
    
    kind = "audio"
    
    def __init__(self, sample_rate: int = 16000):
        if AIORTC_AVAILABLE:
            super().__init__()
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._running = True
        self._pts = 0
        
        # Frame size (20ms of audio at sample_rate)
        self._frame_samples = sample_rate // 50
    
    async def recv(self) -> "AudioFrame":
        """Provide audio frame to WebRTC (called by aiortc)"""
        try:
            audio_bytes = await asyncio.wait_for(
                self._queue.get(),
                timeout=0.1
            )
            
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            
            if len(audio) < self._frame_samples:
                audio = np.pad(audio, (0, self._frame_samples - len(audio)))
            elif len(audio) > self._frame_samples:
                audio = audio[:self._frame_samples]
            
            frame = AudioFrame(format="s16", layout="mono", samples=len(audio))
            frame.sample_rate = self.sample_rate
            frame.pts = self._pts
            self._pts += len(audio)
            
            frame.planes[0].update(audio.tobytes())
            
            return frame
            
        except asyncio.TimeoutError:
            silence = np.zeros(self._frame_samples, dtype=np.int16)
            frame = AudioFrame(format="s16", layout="mono", samples=len(silence))
            frame.sample_rate = self.sample_rate
            frame.pts = self._pts
            self._pts += len(silence)
            frame.planes[0].update(silence.tobytes())
            return frame
            
        except Exception as e:
            logger.error(f"Error in audio recv: {e}")
            raise
    
    def send_audio(self, audio_bytes: bytes) -> None:
        """Queue audio bytes for sending"""
        try:
            self._queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(audio_bytes)
            except:
                pass
    
    def stop(self) -> None:
        """Stop sending audio"""
        self._running = False
