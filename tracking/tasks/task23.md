# Task 23: Add `get_supported_languages()` to `MultiProviderSTT`

> **Priority:** 🔴 P0 | **Phase:** Voice Fix | **Effort:** 1 hour
> **Files:** `src/voice/stt.py`
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

`GET /api/v1/voice/languages` always returns HTTP 500 because `MultiProviderSTT` has no `get_supported_languages()` method, but the endpoint calls it. The frontend Tools Inspector tab cannot populate the language tables.

---

## 🏗️ Implementation Spec

### Add to `MultiProviderSTT` in `src/voice/stt.py`

```python
def get_supported_languages(self) -> list[str]:
    """Get supported language codes from first available provider."""
    for provider in self._providers:
        if hasattr(provider, "get_supported_languages"):
            return provider.get_supported_languages()
    # Fallback: return all known CropFresh languages
    return [lang.value for lang in SupportedLanguage if lang != SupportedLanguage.AUTO]
```

Add after `get_available_providers()` method (around line 595).

---

## ✅ Acceptance Criteria

| #   | Criterion                                        | Weight |
| --- | ------------------------------------------------ | ------ |
| 1   | `GET /api/v1/voice/languages` returns HTTP 200   | 40%    |
| 2   | Response contains non-empty `stt_languages` list | 30%    |
| 3   | Response contains non-empty `tts_languages` list | 30%    |
