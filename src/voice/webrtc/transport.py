"""
WebRTC Transport
================
Core transport manager for bidirectional audio streaming.
"""

import asyncio
import json
import uuid
from typing import AsyncIterator, Callable, Optional

from loguru import logger

from .models import ConnectionState, WebRTCConfig, AudioChunk
from .tracks import AudioReceiveTrack, AudioSendTrack, AIORTC_AVAILABLE

if AIORTC_AVAILABLE:
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, RTCIceCandidate


class WebRTCTransport:
    """
    WebRTC transport for bidirectional audio streaming.
    Manages peer connections, ICE negotiation, and audio tracks.
    """
    
    def __init__(self, config: Optional[WebRTCConfig] = None):
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
        return self._on_state_change
    
    @on_state_change.setter
    def on_state_change(self, callback: Callable[[ConnectionState], None]) -> None:
        self._on_state_change = callback
    
    @property
    def on_message(self) -> Optional[Callable[[dict], None]]:
        return self._on_message
    
    @on_message.setter
    def on_message(self, callback: Callable[[dict], None]) -> None:
        self._on_message = callback
    
    async def create_peer_connection(self) -> None:
        """Create and configure peer connection"""
        ice_servers = [
            RTCIceServer(urls=server.get("urls", []))
            for server in self.config.ice_servers
        ]
        
        config = RTCConfiguration(iceServers=ice_servers)
        self._pc = RTCPeerConnection(configuration=config)
        
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
        
        self._send_track = AudioSendTrack(self.config.sample_rate)
        self._pc.addTrack(self._send_track)
        
        self._data_channel = self._pc.createDataChannel(
            "control",
            ordered=True,
        )
        
        logger.info("Peer connection created")
    
    async def create_offer(self) -> dict:
        """Create SDP offer for initiating connection."""
        if not self._pc:
            await self.create_peer_connection()
        
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        
        return {
            "type": offer.type,
            "sdp": offer.sdp,
        }
    
    async def create_answer(self, offer_sdp: dict) -> dict:
        """Create SDP answer in response to offer."""
        if not self._pc:
            await self.create_peer_connection()
        
        offer = RTCSessionDescription(
            sdp=offer_sdp["sdp"],
            type=offer_sdp["type"],
        )
        await self._pc.setRemoteDescription(offer)
        
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
        """Receive audio from remote peer."""
        if not self._receive_track:
            raise RuntimeError("No audio track available")
        
        async for chunk in self._receive_track.get_audio():
            yield chunk
    
    def send_audio(self, audio_bytes: bytes) -> None:
        """Send audio to remote peer."""
        if self._send_track:
            self._send_track.send_audio(audio_bytes)
    
    def send_message(self, message: dict) -> None:
        """Send control message via data channel."""
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
