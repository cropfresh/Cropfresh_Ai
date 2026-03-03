# Task 22: Fix REST Router — Remove Invalid `use_groq` Kwarg

> **Priority:** 🔴 P0 | **Phase:** Voice Fix | **Effort:** 0.5 hours
> **Files:** `src/api/rest/voice.py` (line ~40)
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

`MultiProviderSTT(...)` in `src/api/rest/voice.py` is initialized with `use_groq=True`, which is not a valid parameter. The `MultiProviderSTT.__init__` signature only accepts `(use_faster_whisper, use_indicconformer, faster_whisper_model)`. This causes every call to `/api/v1/voice/process` to crash with `TypeError: unexpected keyword argument 'use_groq'`.

---

## 🏗️ Implementation Spec

### Fix in `src/api/rest/voice.py`

```diff
 _stt = MultiProviderSTT(
     use_faster_whisper=True,
     use_indicconformer=False,   # disabled on CPU — no model cached
-    use_groq=True,
     faster_whisper_model="small",
 )
```

---

## ✅ Acceptance Criteria

| #   | Criterion                                              | Weight |
| --- | ------------------------------------------------------ | ------ |
| 1   | `MultiProviderSTT` initializes without `TypeError`     | 50%    |
| 2   | `POST /api/v1/voice/process` no longer crashes on init | 50%    |
