# ═══════════════════════════════════════════════
# CropFresh AI — Production Dockerfile
# Multi-stage build:
#   Stage 1 (builder): install deps with uv
#   Stage 2 (runtime): minimal Python slim image
# ═══════════════════════════════════════════════
# Build: docker build -f infra/docker/api.Dockerfile -t cropfresh-ai:latest .
# Run:   docker run -p 8000:8000 --env-file .env cropfresh-ai:latest
# ═══════════════════════════════════════════════

# ─── Stage 1: Dependency builder ─────────────
FROM python:3.11-slim AS builder

# Install uv (fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy dependency manifests first (layer cache optimization)
COPY pyproject.toml uv.lock ./

# Step 1: Create venv
RUN uv venv /build/.venv

# Step 2: Install CPU-only PyTorch into the venv (avoids 5GB CUDA)
RUN /build/.venv/bin/pip install --no-cache-dir --retries 5 --timeout 300 \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install all other deps, skip torch + CUDA (already installed above)
ENV UV_CONCURRENT_DOWNLOADS=2
RUN uv sync --extra voice --no-dev \
    --no-install-package torch \
    --no-install-package torchvision \
    --no-install-package torchaudio \
    --no-install-package nvidia-cublas-cu12 \
    --no-install-package nvidia-cuda-cupti-cu12 \
    --no-install-package nvidia-cuda-nvrtc-cu12 \
    --no-install-package nvidia-cuda-runtime-cu12 \
    --no-install-package nvidia-cudnn-cu12 \
    --no-install-package nvidia-cufft-cu12 \
    --no-install-package nvidia-cufile-cu12 \
    --no-install-package nvidia-curand-cu12 \
    --no-install-package nvidia-cusolver-cu12 \
    --no-install-package nvidia-cusparse-cu12 \
    --no-install-package nvidia-cusparselt-cu12 \
    --no-install-package nvidia-nccl-cu12 \
    --no-install-package nvidia-nvjitlink-cu12 \
    --no-install-package nvidia-nvshmem-cu12 \
    --no-install-package nvidia-nvtx-cu12 \
    --no-install-package triton


# ─── Stage 2: Production runtime ─────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1001 cropfresh && \
    useradd --uid 1001 --gid cropfresh --shell /bin/bash --create-home cropfresh

WORKDIR /app

# Copy installed virtualenv from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code
COPY src/ ./src/
COPY ai/ ./ai/
COPY static/ ./static/

# Ensure virtualenv binaries take priority
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# Switch to non-root user
USER cropfresh

# Expose API port
EXPOSE 8000

# Healthcheck (matches /health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Production server: 2 workers, 30s graceful shutdown
CMD ["uvicorn", "src.api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--timeout-graceful-shutdown", "30", \
    "--log-level", "info"]
