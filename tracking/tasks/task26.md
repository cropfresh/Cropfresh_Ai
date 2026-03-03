# Task 26: Make `faster-whisper` Primary STT, Disable IndicConformer on CPU

> **Priority:** 🟠 P1 | **Phase:** Voice Fix | **Effort:** 2 hours
> **Files:** `src/voice/stt.py`
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

`MultiProviderSTT` initializes `IndicWhisperSTT` (IndicConformer) as its first provider, which tries to download `ai4bharat/indic-conformer-600m-multilingual` (600MB+) from HuggingFace. This fails on CPU without the cached model. `FasterWhisperSTT` is the correct primary provider for dev/CPU environments.

---

## 🏗️ Implementation Spec

### Change default in `MultiProviderSTT.__init__`

```python
def __init__(
    self,
    use_faster_whisper: bool = True,       # primary on CPU
    use_indicconformer: bool = False,      # disabled by default (needs GPU + cached model)
    faster_whisper_model: str = "small",
):
```

### Provider priority order (inside `_initialize_providers`)

```python
# Priority 1: faster-whisper (local, CPU-friendly)
if use_faster_whisper:
    try:
        self._providers.append(FasterWhisperSTT(model_size=faster_whisper_model))
    except Exception as e:
        logger.warning(f"FasterWhisper unavailable: {e}")

# Priority 2: IndicConformer (GPU, AI4Bharat model)
if use_indicconformer:
    try:
        self._providers.append(IndicWhisperSTT())
    except Exception as e:
        logger.warning(f"IndicConformer unavailable: {e}")

# Priority 3: GroqWhisper (cloud fallback, added in Task 29)
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                                                 | Weight |
| --- | ------------------------------------------------------------------------- | ------ |
| 1   | `MultiProviderSTT()` default initializes with faster-whisper only         | 30%    |
| 2   | `get_available_providers()` returns `["faster_whisper"]` on CPU           | 30%    |
| 3   | `transcribe(audio_bytes, "en")` returns a non-empty `TranscriptionResult` | 40%    |
