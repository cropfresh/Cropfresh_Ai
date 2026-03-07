# ═══════════════════════════════════════════════
# CropFresh AI — Production Dockerfile
# Multi-stage build:
#   Stage 1 (builder): install deps with uv
#   Stage 2 (runtime): minimal Python slim image
# ═══════════════════════════════════════════════

FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml uv.lock ./

# Step 1: Install CPU-only PyTorch to system site-packages
#   (avoids 5GB CUDA download that uv.lock pins for linux/x86_64)
RUN pip install --no-cache-dir --retries 5 --timeout 300 \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install everything else via uv, skip torch + all CUDA packages
ENV UV_CONCURRENT_DOWNLOADS=4
RUN uv sync --extra voice --no-dev --no-install-project \
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

# Step 3: Copy CPU torch into the venv
RUN cp -r /usr/local/lib/python3.11/site-packages/torch /build/.venv/lib/python3.11/site-packages/ && \
    cp -r /usr/local/lib/python3.11/site-packages/torch-* /build/.venv/lib/python3.11/site-packages/ && \
    cp -r /usr/local/lib/python3.11/site-packages/torchvision /build/.venv/lib/python3.11/site-packages/ && \
    cp -r /usr/local/lib/python3.11/site-packages/torchvision-* /build/.venv/lib/python3.11/site-packages/ && \
    cp -r /usr/local/lib/python3.11/site-packages/torchaudio /build/.venv/lib/python3.11/site-packages/ && \
    cp -r /usr/local/lib/python3.11/site-packages/torchaudio-* /build/.venv/lib/python3.11/site-packages/


# ─── Stage 2: Production runtime ─────────────
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1001 cropfresh && \
    useradd --uid 1001 --gid cropfresh --shell /bin/bash --create-home cropfresh

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv
COPY src/ ./src/
COPY ai/ ./ai/
COPY static/ ./static/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

USER cropfresh
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--timeout-graceful-shutdown", "30", \
    "--log-level", "info"]
