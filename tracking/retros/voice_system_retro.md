# Sprint Retrospective — Voice Subsystem

## 🟢 What Went Well

- **Multi-Provider Fallbacks**: Excellent robust design in `MultiProviderSTT` that elegantly cascades from local `FasterWhisper` → `IndicConformer` → cloud `GroqWhisper` depending on hardware availability and network limits.
- **Full-Duplex Orchestration**: `DuplexPipeline` successfully implements advanced conversational UX features: speculative TTS (streaming synthesized audio sentence-by-sentence as the LLM generates them) and true barge-in (instantly cancelling TTS/LLM generation when the user interrupts).
- **Edge TTS Integration**: Successfully added Microsoft Edge TTS as a drop-in, zero-cost fallback for IndicTTS with built-in retry mechanisms for transient network failures.
- **Dynamic Language Detection**: The STT pipeline cleanly handles "auto" language detection and propagates the detected language automatically through to the LLM and TTS layers without user intervention.

## 🟡 What Could Improve

- **Hardcoded Voice Selections**: In `DuplexPipeline._synthesize_sentence`, the voice parameter is hardcoded to `"female"` instead of being configurable per-user or per-persona.
- **Audio Decoding Fragility**: `IndicWhisperSTT` relies on complex audio backend acrobatics (trying `soundfile` then falling back to `torchaudio.set_audio_backend("soundfile")`), which is prone to environment-specific crashes on Windows vs Linux.
- **TTS Fallback Bug**: If streaming TTS fails in `DuplexPipeline`, the fallback `_tts.synthesize_full` assumes an MP3 output at 24000Hz unconditionally, bypassing the actual format returned by the TTS provider.

## 🔴 Action Items

- [ ] Refactor `DuplexPipeline` to accept voice preferences (gender/voice ID) dynamically during initialization or via the `process_text` parameters.
- [ ] Centralize audio format decoding/normalisation using `ffmpeg-python` or `pydub` exclusively in `audio_utils.py`, removing the fragile torchaudio backend switching from the STT modules.
- [ ] Fix the TTS streaming fallback block in `DuplexPipeline._synthesize_sentence` to dynamically read the `format` and `sample_rate` from the `full_audio` result instead of hardcoding MP3/24000Hz.
