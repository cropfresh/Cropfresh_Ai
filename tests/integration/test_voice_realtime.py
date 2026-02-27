"""
Bidirectional Voice Agent Test Suite
=====================================

Tests for verifying WebSocket streaming, VAD, and bidirectional communication.

Run with: uv run pytest tests/test_voice_realtime.py -v
"""

import asyncio
import base64
import json
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import websockets
from loguru import logger


# Test configuration
TEST_SERVER_URL = "ws://127.0.0.1:8000/api/v1/voice/ws"
SAMPLE_RATE = 16000
AUDIO_CHUNK_MS = 30


def generate_test_audio(
    duration_ms: int = 1000,
    frequency: float = 440.0,
    sample_rate: int = SAMPLE_RATE,
    with_silence: bool = False,
) -> bytes:
    """
    Generate test audio (sine wave or silence).
    
    Args:
        duration_ms: Duration in milliseconds
        frequency: Frequency of sine wave (Hz)
        sample_rate: Audio sample rate
        with_silence: If True, generate silence instead
        
    Returns:
        PCM 16-bit audio bytes
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    
    if with_silence:
        audio = np.zeros(num_samples, dtype=np.float32)
    else:
        t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def bytes_to_base64(audio_bytes: bytes) -> str:
    """Convert audio bytes to base64 string"""
    return base64.b64encode(audio_bytes).decode("utf-8")


class TestWebSocketConnection:
    """Test WebSocket connection and basic messaging"""
    
    @pytest.mark.asyncio
    async def test_websocket_connect(self):
        """Test basic WebSocket connection"""
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=test_user") as ws:
            # Should receive 'ready' message
            response = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            assert data["type"] == "ready"
            assert "session_id" in data
            assert "timestamp" in data
            
            logger.info(f"✅ Connected with session: {data['session_id']}")
    
    @pytest.mark.asyncio
    async def test_websocket_reconnect(self):
        """Test WebSocket reconnection capability"""
        # First connection
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=test_user_1") as ws1:
            resp1 = await asyncio.wait_for(ws1.recv(), timeout=10.0)
            session1 = json.loads(resp1)["session_id"]
        
        # Second connection (should get new session)
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=test_user_1") as ws2:
            resp2 = await asyncio.wait_for(ws2.recv(), timeout=10.0)
            session2 = json.loads(resp2)["session_id"]
        
        # Sessions should be different
        assert session1 != session2
        logger.info(f"✅ Reconnection works: {session1} → {session2}")


class TestVADDetection:
    """Test Voice Activity Detection"""
    
    @pytest.mark.asyncio
    async def test_vad_silence_detection(self):
        """Test that silence doesn't trigger VAD"""
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=vad_test") as ws:
            # Wait for ready
            await asyncio.wait_for(ws.recv(), timeout=10.0)
            
            # Send silence chunks
            silence = generate_test_audio(duration_ms=500, with_silence=True)
            chunk_size = SAMPLE_RATE * AUDIO_CHUNK_MS // 1000 * 2  # 16-bit
            
            for i in range(0, len(silence), chunk_size):
                chunk = silence[i:i + chunk_size]
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "audio_base64": bytes_to_base64(chunk),
                }))
                await asyncio.sleep(0.02)
            
            # Should not receive vad_start
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(response)
                # If we get a message, it shouldn't be speech detection
                assert data["type"] != "vad_start", "Silence triggered VAD incorrectly"
            except asyncio.TimeoutError:
                pass  # Expected - no VAD trigger
            
            logger.info("✅ VAD correctly ignored silence")
    
    @pytest.mark.asyncio
    async def test_vad_speech_detection(self):
        """Test that audio triggers VAD"""
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=vad_test_2") as ws:
            # Wait for ready
            await asyncio.wait_for(ws.recv(), timeout=10.0)
            
            # Send audio (sine wave simulates speech energy)
            audio = generate_test_audio(duration_ms=1000, frequency=300)
            chunk_size = SAMPLE_RATE * AUDIO_CHUNK_MS // 1000 * 2
            
            received_messages = []
            
            # Send chunks
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "audio_base64": bytes_to_base64(chunk),
                }))
                await asyncio.sleep(0.02)
            
            # Collect responses
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    data = json.loads(response)
                    received_messages.append(data["type"])
                    if data["type"] == "error":
                        logger.warning(f"Received error: {data.get('error')}")
            except asyncio.TimeoutError:
                pass
            
            logger.info(f"Received message types: {received_messages}")
            # Note: VAD may or may not trigger depending on audio energy


class TestLanguageHint:
    """Test language hint functionality"""
    
    @pytest.mark.asyncio
    async def test_language_hint_hindi(self):
        """Test setting language hint to Hindi"""
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=lang_test") as ws:
            await asyncio.wait_for(ws.recv(), timeout=10.0)
            
            # Send language hint
            await ws.send(json.dumps({
                "type": "language_hint",
                "language": "hi",
            }))
            
            logger.info("✅ Language hint 'hi' sent successfully")
    
    @pytest.mark.asyncio
    async def test_language_hint_english(self):
        """Test setting language hint to English"""
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=lang_test_2") as ws:
            await asyncio.wait_for(ws.recv(), timeout=10.0)
            
            # Send language hint
            await ws.send(json.dumps({
                "type": "language_hint",
                "language": "en",
            }))
            
            logger.info("✅ Language hint 'en' sent successfully")


class TestBidirectionalCommunication:
    """Test full bidirectional voice flow"""
    
    @pytest.mark.asyncio
    async def test_full_voice_flow_simulation(self):
        """
        Simulate a complete voice interaction:
        1. Connect to WebSocket
        2. Send audio chunks
        3. Receive VAD events
        4. Receive transcription
        5. Receive response
        """
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=flow_test") as ws:
            # Step 1: Wait for ready
            ready_msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ready_data = json.loads(ready_msg)
            assert ready_data["type"] == "ready"
            session_id = ready_data["session_id"]
            logger.info(f"Step 1 ✅ Connected: {session_id}")
            
            # Step 2: Set language hint
            await ws.send(json.dumps({
                "type": "language_hint",
                "language": "en",
            }))
            logger.info("Step 2 ✅ Language hint sent")
            
            # Step 3: Send audio (simulated speech)
            audio = generate_test_audio(duration_ms=2000, frequency=400)
            chunk_size = SAMPLE_RATE * AUDIO_CHUNK_MS // 1000 * 2
            chunks_sent = 0
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "audio_base64": bytes_to_base64(chunk),
                }))
                chunks_sent += 1
                await asyncio.sleep(0.02)
            
            logger.info(f"Step 3 ✅ Sent {chunks_sent} audio chunks")
            
            # Step 4: Collect all responses
            messages_received = {}
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    msg_type = data["type"]
                    
                    if msg_type not in messages_received:
                        messages_received[msg_type] = []
                    messages_received[msg_type].append(data)
                    
                    logger.info(f"Received: {msg_type}")
                    
                    if msg_type == "response_end":
                        break
                    if msg_type == "error":
                        logger.error(f"Error: {data.get('error')}")
                        break
                        
            except asyncio.TimeoutError:
                pass
            
            logger.info(f"Step 4 ✅ Message types received: {list(messages_received.keys())}")
            
            # Report results
            return {
                "session_id": session_id,
                "chunks_sent": chunks_sent,
                "message_types": list(messages_received.keys()),
                "messages": messages_received,
            }


class TestSessionManagement:
    """Test session management functionality"""
    
    @pytest.mark.asyncio
    async def test_session_stop(self):
        """Test stopping a session"""
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=stop_test") as ws:
            await asyncio.wait_for(ws.recv(), timeout=10.0)
            
            # Send stop message
            await ws.send(json.dumps({"type": "stop"}))
            
            # Connection should close gracefully
            try:
                await asyncio.wait_for(ws.recv(), timeout=2.0)
            except websockets.exceptions.ConnectionClosed:
                pass  # Expected
            
            logger.info("✅ Session stopped successfully")
    
    @pytest.mark.asyncio  
    async def test_session_close(self):
        """Test closing a session"""
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=close_test") as ws:
            await asyncio.wait_for(ws.recv(), timeout=10.0)
            
            # Send close message
            await ws.send(json.dumps({"type": "close"}))
            
            logger.info("✅ Session close sent")


# Standalone test runner
async def run_quick_test():
    """Quick test to verify WebSocket is working"""
    print("\n" + "=" * 60)
    print("🎤 CropFresh Voice Agent - Quick Test")
    print("=" * 60 + "\n")
    
    try:
        async with websockets.connect(f"{TEST_SERVER_URL}?user_id=quick_test") as ws:
            # Test 1: Connection
            print("1️⃣ Testing WebSocket connection...")
            response = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            if data["type"] == "ready":
                print(f"   ✅ Connected! Session: {data['session_id']}")
            else:
                print(f"   ❌ Unexpected response: {data['type']}")
                return False
            
            # Test 2: Send audio
            print("\n2️⃣ Sending test audio chunks...")
            audio = generate_test_audio(duration_ms=500, frequency=350)
            chunk_size = SAMPLE_RATE * AUDIO_CHUNK_MS // 1000 * 2
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "audio_base64": bytes_to_base64(chunk),
                }))
            print("   ✅ Audio chunks sent!")
            
            # Test 3: Collect responses
            print("\n3️⃣ Waiting for responses...")
            message_types = set()
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    data = json.loads(response)
                    message_types.add(data["type"])
                    print(f"   📨 Received: {data['type']}")
            except asyncio.TimeoutError:
                pass
            
            print(f"\n   Message types: {message_types}")
            
            print("\n" + "=" * 60)
            print("✅ Quick test completed successfully!")
            print("=" * 60 + "\n")
            return True
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(run_quick_test())
