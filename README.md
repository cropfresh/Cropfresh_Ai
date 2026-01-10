# CropFresh AI Service

AI-powered backend for agricultural marketplace with voice, vision, and RAG capabilities.

## 🚀 Quick Start (UV)

```bash
# 1. Create & activate virtual environment
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
uv sync --extra voice

# 3. Set environment variables
copy .env.example .env
# Edit .env with your GROQ_API_KEY

# 4. Run the service
uv run uvicorn src.api.main:app --reload
```

Visit: http://localhost:8000/docs

---

## 📦 Package Manager

**This project uses UV** - a fast Python package manager (10-100x faster than pip).

```bash
# Install UV (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install UV (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 🔧 Installation Options

| Command | What it installs |
|---------|-----------------|
| `uv sync` | Core (FastAPI, LangGraph, Groq) |
| `uv sync --extra voice` | + Voice (STT/TTS) |
| `uv sync --extra ml` | + ML models (PyTorch) |
| `uv sync --extra vision` | + Vision (YOLOv11) |
| `uv sync --all-extras` | Everything |

---

## 📁 Project Structure

```
cropfresh-service-ai/
├── src/
│   ├── agents/          # AI Agents
│   │   ├── voice_agent.py
│   │   ├── pricing_agent.py
│   │   └── knowledge_agent.py
│   ├── voice/           # Voice Module
│   │   ├── stt.py       # Speech-to-Text
│   │   ├── tts.py       # Text-to-Speech
│   │   └── entity_extractor.py
│   ├── api/             # API Layer
│   │   ├── main.py
│   │   ├── rest/
│   │   └── websocket.py
│   ├── rag/             # RAG System
│   └── config/
├── tests/
├── WORKFLOW_STATUS.md   # Track changes
├── pyproject.toml
└── .env.example
```

---

## 🎤 Voice API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/voice/process` | POST | Voice-in → Voice-out |
| `/api/v1/voice/transcribe` | POST | Audio → Text |
| `/api/v1/voice/synthesize` | POST | Text → Audio |
| `/ws/voice/{user_id}` | WS | Real-time |

### Supported Languages
Hindi, Kannada, Telugu, Tamil, Malayalam, Marathi, Gujarati, Bengali, Punjabi, Odia, English

---

## 🧪 Development Commands

```bash
# Run tests
uv run pytest

# Type check
uv run mypy src/

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

---

## 📋 Status

See [WORKFLOW_STATUS.md](./WORKFLOW_STATUS.md) for:
- Current progress
- File change log
- Pending tasks
- Setup instructions

---

## ⚙️ Environment Variables

```env
# Required
GROQ_API_KEY=gsk_xxxxx

# Optional
QDRANT_HOST=localhost
QDRANT_PORT=6333
LLM_PROVIDER=groq
DEBUG=true
```

---

## 📜 License

MIT
