import asyncio
import logging
from src.agents.voice_agent import VoiceAgent
from src.voice.entity_extractor import VoiceEntityExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)

async def test_memory():
    print("\n--- Testing Voice Agent Memory Flow ---")
    
    # Initialize basic components (no real services or LLM needed for template flow testing)
    extractor = VoiceEntityExtractor(llm_provider=None)
    # Stub STT to just return the text we pass in to skip audio processing
    class StubSTT:
        async def transcribe(self, audio, language):
            from src.voice.stt import TranscriptionResult
            return TranscriptionResult(text=audio.decode('utf-8'), language="en", confidence=0.99)
            
    # Stub TTS to do nothing
    class StubTTS:
        async def synthesize(self, text, language):
            from src.voice.tts import SynthesisResult
            return SynthesisResult(audio=b"fake_audio", text=text, language=language)

    agent = VoiceAgent(
        stt=StubSTT(),
        tts=StubTTS(),
        entity_extractor=extractor
    )
    
    session_id = "test_memory_session"
    user_id = "farmer_123"
    
    # Turn 1
    t1 = "Can you help me to list the tomatoes?"
    print(f"\nUser: {t1}")
    r1 = await agent.process_voice(t1.encode('utf-8'), user_id, session_id, language="en")
    print(f"Bot : {r1.response_text}")
    print(f"Entities: {r1.entities}")
    print(f"Intent  : {r1.intent}")
    
    # Turn 2
    t2 = "20 kgs"
    print(f"\nUser: {t2}")
    r2 = await agent.process_voice(t2.encode('utf-8'), user_id, session_id, language="en")
    print(f"Bot : {r2.response_text}")
    print(f"Entities: {r2.entities}")
    print(f"Intent  : {r2.intent}")

    # Turn 3
    t3 = "20 rupees per kg"
    print(f"\nUser: {t3}")
    r3 = await agent.process_voice(t3.encode('utf-8'), user_id, session_id, language="en")
    print(f"Bot : {r3.response_text}")
    print(f"Entities: {r3.entities}")
    print(f"Intent  : {r3.intent}")
    
    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(test_memory())
