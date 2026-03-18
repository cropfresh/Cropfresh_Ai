FROM python:3.11-slim

WORKDIR /app/services/vad-service

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    grpcio \
    grpcio-tools \
    loguru \
    numpy \
    onnxruntime \
    pydantic \
    pydantic-settings

COPY services/vad-service ./services/vad-service
COPY src/shared /app/src/shared

ENV PYTHONPATH="/app/services/vad-service:/app"

EXPOSE 8101 50061

CMD ["python", "-m", "app.run_service"]
