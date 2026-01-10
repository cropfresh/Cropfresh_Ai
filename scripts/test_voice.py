"""
Voice Agent Test Script
======================
Simple script to test the Voice Agent components without needing ML models.
Uses Edge TTS (free, no API key needed) for text-to-speech.

Usage:
    uv run python scripts/test_voice.py
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_tts():
    """Test Text-to-Speech with Edge TTS"""
    print("\n" + "="*60)
    print("🔊 Testing Text-to-Speech (Edge TTS)")
    print("="*60)
    
    from src.voice.tts import IndicTTS
    
    tts = IndicTTS(use_edge_fallback=True)
    
    # Test Hindi
    print("\n📝 Testing Hindi TTS...")
    result = await tts.synthesize(
        text="नमस्ते, मैं क्रॉपफ्रेश हूं। आज टमाटर का भाव 25 रुपये प्रति किलो है।",
        language="hi"
    )
    
    if result.is_successful:
        print(f"   ✅ Hindi TTS successful!")
        print(f"   Audio size: {len(result.audio)} bytes")
        print(f"   Duration: ~{result.duration_seconds:.1f}s")
        print(f"   Provider: {result.provider}")
        
        # Save audio file
        output_file = "test_hindi_output.wav"
        with open(output_file, "wb") as f:
            f.write(result.audio)
        print(f"   📁 Saved to: {output_file}")
    else:
        print(f"   ❌ Hindi TTS failed")
    
    # Test Kannada
    print("\n📝 Testing Kannada TTS...")
    result_kn = await tts.synthesize(
        text="ನಮಸ್ಕಾರ, ನಾನು ಕ್ರಾಪ್ಫ್ರೆಶ್. ಇಂದು ಟೊಮೆಟೊ ಬೆಲೆ 25 ರೂಪಾಯಿ.",
        language="kn"
    )
    
    if result_kn.is_successful:
        print(f"   ✅ Kannada TTS successful!")
        print(f"   Audio size: {len(result_kn.audio)} bytes")
        print(f"   Provider: {result_kn.provider}")
        
        with open("test_kannada_output.wav", "wb") as f:
            f.write(result_kn.audio)
        print(f"   📁 Saved to: test_kannada_output.wav")
    else:
        print(f"   ❌ Kannada TTS failed")
    
    # Test English
    print("\n📝 Testing English TTS...")
    result_en = await tts.synthesize(
        text="Hello! I am CropFresh. Today's tomato price is 25 rupees per kg.",
        language="en"
    )
    
    if result_en.is_successful:
        print(f"   ✅ English TTS successful!")
        print(f"   Audio size: {len(result_en.audio)} bytes")
        
        with open("test_english_output.wav", "wb") as f:
            f.write(result_en.audio)
        print(f"   📁 Saved to: test_english_output.wav")
    else:
        print(f"   ❌ English TTS failed")
    
    return result.is_successful


async def test_entity_extraction():
    """Test Entity Extraction"""
    print("\n" + "="*60)
    print("🧠 Testing Entity Extraction")
    print("="*60)
    
    from src.voice.entity_extractor import VoiceEntityExtractor, VoiceIntent
    
    extractor = VoiceEntityExtractor()
    
    test_cases = [
        ("मेरे पास 200 किलो टमाटर है", "hi", VoiceIntent.CREATE_LISTING),
        ("टमाटर का भाव क्या है कोलार में", "hi", VoiceIntent.CHECK_PRICE),
        ("नमस्ते", "hi", VoiceIntent.GREETING),
        ("मदद करो", "hi", VoiceIntent.HELP),
        ("I have 100 kg potatoes", "en", VoiceIntent.CREATE_LISTING),
        ("What is the price of onion", "en", VoiceIntent.CHECK_PRICE),
    ]
    
    passed = 0
    for text, lang, expected_intent in test_cases:
        result = await extractor.extract(text, lang, use_llm=False)
        
        status = "✅" if result.intent == expected_intent else "❌"
        print(f"\n{status} Text: \"{text}\"")
        print(f"   Language: {lang}")
        print(f"   Expected: {expected_intent.value}")
        print(f"   Got: {result.intent.value}")
        print(f"   Entities: {result.entities}")
        print(f"   Confidence: {result.confidence:.2f}")
        
        if result.intent == expected_intent:
            passed += 1
    
    print(f"\n📊 Entity Extraction: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


async def test_voice_agent_flow():
    """Test complete Voice Agent flow (without actual audio)"""
    print("\n" + "="*60)
    print("🎤 Testing Voice Agent Flow (Mock)")
    print("="*60)
    
    from src.voice.entity_extractor import VoiceEntityExtractor, VoiceIntent
    from src.voice.tts import IndicTTS
    
    # Simulate the flow
    extractor = VoiceEntityExtractor()
    tts = IndicTTS(use_edge_fallback=True)
    
    # Simulate: Farmer says "मेरे पास 100 किलो टमाटर है"
    user_text = "मेरे पास 100 किलो टमाटर है"
    print(f"\n👨‍🌾 Farmer says: \"{user_text}\"")
    
    # Step 1: Extract intent
    extraction = await extractor.extract(user_text, "hi", use_llm=False)
    print(f"\n📝 Intent: {extraction.intent.value}")
    print(f"   Entities: {extraction.entities}")
    
    # Step 2: Generate response
    if extraction.intent == VoiceIntent.CREATE_LISTING:
        crop = extraction.entities.get("crop", "सब्जी")
        quantity = extraction.entities.get("quantity", 100)
        unit = extraction.entities.get("unit", "kg")
        response_text = f"आपकी {quantity} {unit} {crop} की लिस्टिंग बन गई है। खरीदार मिलने पर हम आपको बताएंगे।"
    else:
        response_text = "मुझे समझ नहीं आया।"
    
    print(f"\n🤖 Response: \"{response_text}\"")
    
    # Step 3: Synthesize audio
    result = await tts.synthesize(response_text, "hi")
    
    if result.is_successful:
        print(f"\n✅ Voice Agent Flow Complete!")
        print(f"   Audio generated: {len(result.audio)} bytes")
        
        with open("test_flow_output.wav", "wb") as f:
            f.write(result.audio)
        print(f"   📁 Saved response audio to: test_flow_output.wav")
        return True
    else:
        print(f"\n❌ TTS failed")
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 CROPFRESH VOICE AGENT - TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: TTS
    try:
        results["TTS"] = await test_tts()
    except Exception as e:
        print(f"❌ TTS Test Error: {e}")
        results["TTS"] = False
    
    # Test 2: Entity Extraction
    try:
        results["Entity Extraction"] = await test_entity_extraction()
    except Exception as e:
        print(f"❌ Entity Extraction Error: {e}")
        results["Entity Extraction"] = False
    
    # Test 3: Voice Agent Flow
    try:
        results["Voice Agent Flow"] = await test_voice_agent_flow()
    except Exception as e:
        print(f"❌ Voice Agent Flow Error: {e}")
        results["Voice Agent Flow"] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("🎉 All tests passed!" if all_passed else "⚠️ Some tests failed"))
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
