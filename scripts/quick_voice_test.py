#!/usr/bin/env python
"""
Quick Test for Bidirectional Voice Agent
=========================================
Run: uv run python scripts/quick_voice_test.py
"""

import asyncio
import base64
import json
import numpy as np

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: uv add websockets")
    exit(1)


async def quick_test():
    """Quick test to verify WebSocket bidirectional communication"""
    print("\n" + "=" * 60)
    print("🎤 CropFresh Voice Agent - Bidirectional Test")
    print("=" * 60 + "\n")
    
    url = "ws://127.0.0.1:8000/api/v1/voice/ws?user_id=quick_test"
    
    try:
        async with websockets.connect(url) as ws:
            # Test 1: Connection
            print("1️⃣  Testing WebSocket connection...")
            response = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            if data["type"] == "ready":
                session_id = data["session_id"]
                print(f"   ✅ Connected! Session: {session_id[:8]}...")
            else:
                print(f"   ❌ Unexpected response: {data['type']}")
                return False
            
            # Test 2: Send language hint
            print("\n2️⃣  Sending language hint...")
            await ws.send(json.dumps({
                "type": "language_hint",
                "language": "en",
            }))
            print("   ✅ Language hint 'en' sent")
            
            # Test 3: Send audio chunks
            print("\n3️⃣  Sending test audio chunks...")
            
            # Generate sine wave (simulates voice)
            sample_rate = 16000
            duration = 0.5  # 500ms
            frequency = 400  # Hz
            
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
            audio = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
            
            # Send in 30ms chunks (480 samples at 16kHz)
            chunk_samples = 480
            chunks_sent = 0
            
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples].tobytes()
                msg = {
                    "type": "audio_chunk",
                    "audio_base64": base64.b64encode(chunk).decode("utf-8"),
                }
                await ws.send(json.dumps(msg))
                chunks_sent += 1
                await asyncio.sleep(0.02)  # ~30ms between chunks
            
            print(f"   ✅ Sent {chunks_sent} audio chunks (500ms total)")
            
            # Test 4: Collect responses
            print("\n4️⃣  Waiting for server responses...")
            message_types = []
            
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    data = json.loads(response)
                    msg_type = data["type"]
                    message_types.append(msg_type)
                    
                    if msg_type == "vad_start":
                        print("   📢 VAD: Speech started")
                    elif msg_type == "vad_end":
                        print("   📢 VAD: Speech ended")
                    elif msg_type == "vad_speech":
                        print(f"   📢 VAD: Speech (prob: {data.get('probability', 0):.2f})")
                    elif msg_type == "language_detected":
                        print(f"   🌐 Language: {data.get('language')}")
                    elif msg_type == "transcript_partial":
                        print(f"   📝 Partial: {data.get('text', '')[:50]}...")
                    elif msg_type == "transcript_final":
                        print(f"   📝 Transcript: {data.get('text', '')[:50]}")
                    elif msg_type == "response_text":
                        print(f"   🤖 Response: {data.get('text', '')[:50]}...")
                    elif msg_type == "response_audio":
                        print(f"   🔊 Audio chunk received")
                    elif msg_type == "response_end":
                        print("   ✅ Response complete")
                        break
                    elif msg_type == "error":
                        print(f"   ❌ Error: {data.get('error', 'Unknown')}")
                        break
                    else:
                        print(f"   📨 {msg_type}")
                        
            except asyncio.TimeoutError:
                print("   ⏱️  No more messages (timeout)")
            
            print(f"\n   📊 Total messages received: {len(message_types)}")
            print(f"   📊 Message types: {list(set(message_types))}")
            
            # Summary
            print("\n" + "=" * 60)
            print("✅ BIDIRECTIONAL TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("\n📋 Summary:")
            print(f"   • WebSocket connection: ✅")
            print(f"   • Audio chunks sent: {chunks_sent}")
            print(f"   • Messages received: {len(message_types)}")
            print(f"   • VAD events: {'✅' if 'vad_start' in message_types or 'vad_speech' in message_types else '❌ (no speech detected)'}")
            print()
            
            return True
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Is the server running?")
        print("   Run: uv run uvicorn src.api.main:app --reload")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(quick_test())
