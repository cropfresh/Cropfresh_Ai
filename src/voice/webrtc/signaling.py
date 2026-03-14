"""
WebRTC Signaling
================
WebSocket signaling wrapper for WebRTC connection establishment.
"""

from typing import Optional
from loguru import logger

from .transport import WebRTCTransport


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
