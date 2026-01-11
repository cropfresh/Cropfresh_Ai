"""
WebRTC Transport Layer for CropFresh Voice Agent

Uses aiortc for Python-native WebRTC implementation.
Provides bidirectional real-time audio streaming.

Features:
- Peer-to-peer audio streaming with <100ms latency
- ICE candidate exchange via WebSocket signaling
- Opus codec for high-quality audio
- Data channel for control messages
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Optional

import numpy as np
from loguru import logger

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
    from aiortc.contrib.media import MediaRecorder, MediaPlayer
    from aiortc.mediastreams import MediaStreamTrack, AudioStreamTrack
    from av import AudioFrame
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    logger.warning("aiortc not installed. WebRTC features disabled.")


class ConnectionState(Enum):
    """WebRTC connection states"""
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class WebRTCConfig:
    """WebRTC configuration"""
    ice_servers: list[dict] = field(default_factory=lambda: [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ])
    audio_codec: str = "opus"
    sample_rate: int = 16000
    channels: int = 1
    enable_dtx: bool = True  # Discontinuous transmission


@dataclass
class AudioChunk:
    """Audio chunk with metadata"""
    data: bytes
    sample_rate: int
    channels: int
    timestamp_ms: float
    samples: int


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
        # This is called when we need to provide audio to the remote peer
        # For receiving, we use the on_frame callback instead
        frame = AudioFrame(format="s16", layout="mono", samples=480)
        frame.sample_rate = self.sample_rate
        return frame
    
    def on_frame(self, frame: "AudioFrame") -> None:
        """Handle incoming audio frame"""
        try:
            # Convert frame to bytes
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
            
            # Non-blocking put
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                # Drop oldest chunk
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
        self._frame_samples = sample_rate // 50  # 20ms
    
    async def recv(self) -> "AudioFrame":
        """Provide audio frame to WebRTC (called by aiortc)"""
        try:
            # Wait for audio data
            audio_bytes = await asyncio.wait_for(
                self._queue.get(),
                timeout=0.1
            )
            
            # Convert bytes to frame
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Ensure correct frame size
            if len(audio) < self._frame_samples:
                audio = np.pad(audio, (0, self._frame_samples - len(audio)))
            elif len(audio) > self._frame_samples:
                audio = audio[:self._frame_samples]
            
            # Create audio frame
            frame = AudioFrame(format="s16", layout="mono", samples=len(audio))
            frame.sample_rate = self.sample_rate
            frame.pts = self._pts
            self._pts += len(audio)
            
            # Copy data
            frame.planes[0].update(audio.tobytes())
            
            return frame
            
        except asyncio.TimeoutError:
            # Return silence if no audio available
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
            # Drop oldest
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(audio_bytes)
            except:
                pass
    
    def stop(self) -> None:
        """Stop sending audio"""
        self._running = False


class WebRTCTransport:
    """
    WebRTC transport for bidirectional audio streaming.
    
    Manages peer connections, ICE negotiation, and audio tracks.
    
    Usage:
        transport = WebRTCTransport()
        
        # Handle signaling messages from client
        offer = await websocket.recv()
        answer = await transport.create_answer(offer)
        await websocket.send(answer)
        
        # Receive audio
        async for chunk in transport.receive_audio():
            process_audio(chunk)
        
        # Send audio
        transport.send_audio(audio_bytes)
    """
    
    def __init__(self, config: Optional[WebRTCConfig] = None):
        """
        Initialize WebRTC transport.
        
        Args:
            config: WebRTC configuration
        """
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc not installed. Run: pip install aiortc")
        
        self.config = config or WebRTCConfig()
        self.connection_id = str(uuid.uuid4())
        
        self._pc: Optional[RTCPeerConnection] = None
        self._receive_track: Optional[AudioReceiveTrack] = None
        self._send_track: Optional[AudioSendTrack] = None
        self._data_channel = None
        
        self._state = ConnectionState.NEW
        self._on_state_change: Optional[Callable[[ConnectionState], None]] = None
        self._on_message: Optional[Callable[[dict], None]] = None
        
        logger.info(f"WebRTC transport created: {self.connection_id}")
    
    @property
    def state(self) -> ConnectionState:
        """Current connection state"""
        return self._state
    
    @property
    def on_state_change(self) -> Optional[Callable[[ConnectionState], None]]:
        """Callback for connection state changes"""
        return self._on_state_change
    
    @on_state_change.setter
    def on_state_change(self, callback: Callable[[ConnectionState], None]) -> None:
        self._on_state_change = callback
    
    @property
    def on_message(self) -> Optional[Callable[[dict], None]]:
        """Callback for data channel messages"""
        return self._on_message
    
    @on_message.setter
    def on_message(self, callback: Callable[[dict], None]) -> None:
        self._on_message = callback
    
    async def create_peer_connection(self) -> None:
        """Create and configure peer connection"""
        # Create peer connection with ICE servers
        from aiortc import RTCConfiguration, RTCIceServer
        
        ice_servers = [
            RTCIceServer(urls=server.get("urls", []))
            for server in self.config.ice_servers
        ]
        
        config = RTCConfiguration(iceServers=ice_servers)
        self._pc = RTCPeerConnection(configuration=config)
        
        # Set up event handlers
        @self._pc.on("connectionstatechange")
        async def on_connection_state_change():
            state_map = {
                "new": ConnectionState.NEW,
                "connecting": ConnectionState.CONNECTING,
                "connected": ConnectionState.CONNECTED,
                "disconnected": ConnectionState.DISCONNECTED,
                "failed": ConnectionState.FAILED,
                "closed": ConnectionState.CLOSED,
            }
            self._state = state_map.get(
                self._pc.connectionState,
                ConnectionState.NEW
            )
            logger.info(f"WebRTC connection state: {self._state.value}")
            
            if self._on_state_change:
                self._on_state_change(self._state)
        
        @self._pc.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "audio":
                self._receive_track = AudioReceiveTrack(self.config.sample_rate)
                
                @track.on("ended")
                async def on_ended():
                    logger.info("Audio track ended")
                    if self._receive_track:
                        self._receive_track.stop()
        
        @self._pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel received: {channel.label}")
            self._data_channel = channel
            
            @channel.on("message")
            def on_message(message):
                try:
                    data = json.loads(message)
                    if self._on_message:
                        self._on_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message}")
        
        # Create send track
        self._send_track = AudioSendTrack(self.config.sample_rate)
        self._pc.addTrack(self._send_track)
        
        # Create data channel for control messages
        self._data_channel = self._pc.createDataChannel(
            "control",
            ordered=True,
        )
        
        logger.info("Peer connection created")
    
    async def create_offer(self) -> dict:
        """
        Create SDP offer for initiating connection.
        
        Returns:
            SDP offer as dict
        """
        if not self._pc:
            await self.create_peer_connection()
        
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        
        return {
            "type": offer.type,
            "sdp": offer.sdp,
        }
    
    async def create_answer(self, offer_sdp: dict) -> dict:
        """
        Create SDP answer in response to offer.
        
        Args:
            offer_sdp: Remote SDP offer
            
        Returns:
            SDP answer as dict
        """
        if not self._pc:
            await self.create_peer_connection()
        
        # Set remote description
        offer = RTCSessionDescription(
            sdp=offer_sdp["sdp"],
            type=offer_sdp["type"],
        )
        await self._pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        
        return {
            "type": answer.type,
            "sdp": answer.sdp,
        }
    
    async def set_remote_description(self, sdp: dict) -> None:
        """Set remote SDP description"""
        if not self._pc:
            raise RuntimeError("Peer connection not created")
        
        description = RTCSessionDescription(
            sdp=sdp["sdp"],
            type=sdp["type"],
        )
        await self._pc.setRemoteDescription(description)
    
    async def add_ice_candidate(self, candidate_data: dict) -> None:
        """Add ICE candidate from remote peer"""
        if not self._pc:
            raise RuntimeError("Peer connection not created")
        
        if candidate_data.get("candidate"):
            candidate = RTCIceCandidate(
                sdpMid=candidate_data.get("sdpMid"),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                candidate=candidate_data["candidate"],
            )
            await self._pc.addIceCandidate(candidate)
    
    async def receive_audio(self) -> AsyncIterator[AudioChunk]:
        """
        Receive audio from remote peer.
        
        Yields:
            AudioChunk objects
        """
        if not self._receive_track:
            raise RuntimeError("No audio track available")
        
        async for chunk in self._receive_track.get_audio():
            yield chunk
    
    def send_audio(self, audio_bytes: bytes) -> None:
        """
        Send audio to remote peer.
        
        Args:
            audio_bytes: PCM audio bytes (16-bit, mono)
        """
        if self._send_track:
            self._send_track.send_audio(audio_bytes)
    
    def send_message(self, message: dict) -> None:
        """
        Send control message via data channel.
        
        Args:
            message: JSON-serializable message
        """
        if self._data_channel and self._data_channel.readyState == "open":
            self._data_channel.send(json.dumps(message))
    
    async def close(self) -> None:
        """Close the connection"""
        if self._receive_track:
            self._receive_track.stop()
        
        if self._send_track:
            self._send_track.stop()
        
        if self._pc:
            await self._pc.close()
        
        self._state = ConnectionState.CLOSED
        logger.info(f"WebRTC transport closed: {self.connection_id}")


class WebRTCSignaling:
    """
    WebSocket signaling for WebRTC connection establishment.
    
    Handles SDP offer/answer exchange and ICE candidate relaying.
    """
    
    def __init__(self, transport: WebRTCTransport):
        """
        Initialize signaling.
        
        Args:
            transport: WebRTC transport instance
        """
        self.transport = transport
        self._pending_candidates: list[dict] = []
    
    async def handle_message(self, message: dict) -> Optional[dict]:
        """
        Handle signaling message.
        
        Args:
            message: Signaling message
            
        Returns:
            Response message or None
        """
        msg_type = message.get("type")
        
        if msg_type == "offer":
            # Create answer to offer
            answer = await self.transport.create_answer(message)
            return {"type": "answer", **answer}
        
        elif msg_type == "answer":
            # Set remote answer
            await self.transport.set_remote_description(message)
            return None
        
        elif msg_type == "ice-candidate":
            # Add ICE candidate
            await self.transport.add_ice_candidate(message.get("candidate", {}))
            return None
        
        elif msg_type == "ping":
            return {"type": "pong"}
        
        else:
            logger.warning(f"Unknown signaling message type: {msg_type}")
            return None
