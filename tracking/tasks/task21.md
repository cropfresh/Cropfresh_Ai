# Task 21: Install `faster-whisper` Package

> **Priority:** 🔴 P0 | **Phase:** Voice Fix | **Effort:** 0.5 hours
> **Files:** `pyproject.toml`
> **Status:** [x] Completed — 2026-03-03

---

## 📌 Problem Statement

`faster-whisper` is not installed in the venv. Every STT call crashes with `ModuleNotFoundError`. This is the primary local speech-to-text engine — without it the entire voice pipeline is broken.

---

## 🏗️ Implementation Spec

### 1. Add to pyproject.toml

```toml
[project.dependencies]
faster-whisper = ">=1.0.3"
```

### 2. Run uv sync

```bash
uv add faster-whisper
uv sync
```

### 3. Verify install

```bash
uv run python -c "import faster_whisper; print('faster-whisper OK:', faster_whisper.__version__)"
```

---

## ✅ Acceptance Criteria

| #   | Criterion                              | Weight |
| --- | -------------------------------------- | ------ |
| 1   | `faster-whisper` importable in venv    | 60%    |
| 2   | `FasterWhisperSTT` loads without error | 40%    |
