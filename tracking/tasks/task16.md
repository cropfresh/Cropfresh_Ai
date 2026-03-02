# Task 16: Expand Voice Agent — 10+ Languages & 5 New Intents

> **Priority:** 🟡 P2 | **Phase:** 4 | **Status:** ✅ COMPLETE  
> **Files:** `src/agents/voice_agent.py`, `src/voice/entity_extractor/` (package)  
> **Score:** 9/10 — Zero-literacy-barrier with 10 Indian languages

---

## 📌 Problem Statement

Voice agent currently supports 3 languages (Kannada, Hindi, English) with ~7 intents. Business model requires 10+ languages and comprehensive intent coverage for zero-literacy-barrier access.

---

## 🏗️ Implementation Spec

### Languages to Add

| Language  | Code | Edge TTS Voice       | Region Priority |
| --------- | ---- | -------------------- | --------------- |
| Kannada   | kn   | kn-IN-SapnaNeural    | 🔴 Primary      |
| Hindi     | hi   | hi-IN-SwaraNeural    | 🔴 Primary      |
| English   | en   | en-IN-NeerjaNeural   | 🟠 Secondary    |
| Tamil     | ta   | ta-IN-PallaviNeural  | 🟠 Secondary    |
| Telugu    | te   | te-IN-ShrutiNeural   | 🟠 Secondary    |
| Marathi   | mr   | mr-IN-AarohiNeural   | 🟡 Tertiary     |
| Bengali   | bn   | bn-IN-TanishaaNeural | 🟡 Tertiary     |
| Gujarati  | gu   | gu-IN-DhwaniNeural   | 🟡 Tertiary     |
| Punjabi   | pa   | pa-IN-GurpreetNeural | 🟡 Tertiary     |
| Malayalam | ml   | ml-IN-SobhanaNeural  | 🟡 Tertiary     |

### New Intents

```python
NEW_INTENTS = {
    'find_buyer': {
        'patterns_en': ['find buyer', 'who wants to buy', 'sell my produce'],
        'patterns_kn': ['ಖರೀದಿದಾರ ಹುಡುಕಿ', 'ಯಾರು ಕೊಳ್ಳಲು ಬಯಸುತ್ತಾರೆ'],
        'patterns_hi': ['खरीदार खोजो', 'कौन खरीदना चाहता है'],
    },
    'check_weather': {
        'patterns_en': ['weather', 'will it rain', 'forecast'],
        'patterns_kn': ['ಹವಾಮಾನ', 'ಮಳೆ ಬರುತ್ತಾ'],
        'patterns_hi': ['मौसम', 'बारिश होगी'],
    },
    'get_advisory': {
        'patterns_en': ['farming advice', 'how to grow', 'pest problem'],
        'patterns_kn': ['ಕೃಷಿ ಸಲಹೆ', 'ಬೆಳೆ ಸಲಹೆ'],
        'patterns_hi': ['खेती सलाह', 'कैसे उगाएं'],
    },
    'dispute_status': {
        'patterns_en': ['dispute', 'complaint', 'problem with order'],
        'patterns_kn': ['ದೂರು', 'ಸಮಸ್ಯೆ'],
        'patterns_hi': ['शिकायत', 'विवाद'],
    },
    'weekly_demand': {
        'patterns_en': ['what to grow', 'demand list', 'which crop'],
        'patterns_kn': ['ಏನು ಬೆಳೆಯಬೇಕು', 'ಬೇಡಿಕೆ ಪಟ್ಟಿ'],
        'patterns_hi': ['क्या उगाएं', 'मांग सूची'],
    },
}
```

### Commodity Name Normalization (Regional → Standard)

```python
COMMODITY_ALIASES = {
    # Kannada names → standard
    'ಟೊಮೆಟೊ': 'tomato', 'ಈರುಳ್ಳಿ': 'onion', 'ಆಲೂಗಡ್ಡೆ': 'potato',
    'ಬೀನ್ಸ್': 'beans', 'ಮೆಣಸಿನಕಾಯಿ': 'green_chilli',
    # Hindi names → standard
    'टमाटर': 'tomato', 'प्याज': 'onion', 'आलू': 'potato',
    'मिर्च': 'green_chilli', 'गोभी': 'cauliflower',
    # Tamil names → standard
    'தக்காளி': 'tomato', 'வெங்காயம்': 'onion',
}
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                                  | Weight |
| --- | ---------------------------------------------------------- | ------ |
| 1   | Response templates in 10+ languages                        | 25%    |
| 2   | 5 new intents with multi-language patterns                 | 25%    |
| 3   | Commodity name normalization across languages              | 20%    |
| 4   | Auto-language detection from STT output                    | 15%    |
| 5   | Unit test: templates × intents × languages coverage matrix | 15%    |
