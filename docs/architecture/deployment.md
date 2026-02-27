# Deployment — CropFresh AI

## Platform: GCP Cloud Run

### Architecture
- **Service**: Cloud Run (serverless container)
- **Region**: asia-south1 (Mumbai)
- **Auto-scaling**: 0 → 10 instances
- **Memory**: 2GB per instance
- **CPU**: 2 vCPUs per instance

### Environments
| Env | URL | Auto-deploy |
|-----|-----|-------------|
| Local | localhost:8000 | Manual |
| Staging | cropfresh-staging.run.app | On push to `develop` |
| Production | api.cropfresh.in | On push to `main` |

### CI/CD Pipeline
1. Push to GitHub → GitHub Actions triggered
2. Run lint + tests
3. Build Docker image
4. Push to GCP Container Registry
5. Deploy to Cloud Run
6. Health check verification

### Docker Build
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables (Production)
Managed via GCP Secret Manager. See `.env.example` for full list.

### Monitoring
- Cloud Run metrics (CPU, memory, latency)
- Custom dashboards in `infra/monitoring/dashboards/`
- Alert policies in `infra/monitoring/alerts/`
