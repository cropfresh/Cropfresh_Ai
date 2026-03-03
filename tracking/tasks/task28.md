# Task 28: Pre-download Silero VAD ONNX Model at Startup

> **Priority:** 🟠 P1 | **Phase:** Voice Fix | **Effort:** 1 hour
> **Files:** `src/api/main.py`
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

Silero VAD downloads its ONNX model (~1.8MB) from GitHub on first initialization. If this happens mid-request it adds latency and can fail on slow networks. Pre-downloading at startup ensures it's ready before any WS connection arrives.

---

## 🏗️ Implementation Spec

### In `src/api/main.py` lifespan function

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing startup ...

    # Pre-download Silero VAD model (non-fatal)
    try:
        from src.voice.vad import SileroVAD
        vad = SileroVAD()
        await vad.initialize()
        logger.info("✅ Silero VAD ready")
    except Exception as e:
        logger.warning(f"⚠️ Silero VAD unavailable: {e} — WebSocket VAD will be skipped")

    yield
    # ... existing shutdown ...
```

Model is cached at: `~/.cache/silero_vad/silero_vad.onnx` (1.8MB).

---

## ✅ Acceptance Criteria

| #   | Criterion                                                       | Weight |
| --- | --------------------------------------------------------------- | ------ |
| 1   | Server starts up and logs `✅ Silero VAD ready`                 | 50%    |
| 2   | Server still starts if VAD fails (non-fatal — logs warning)     | 30%    |
| 3   | ONNX model cached at `~/.cache/silero_vad/` after first startup | 20%    |
