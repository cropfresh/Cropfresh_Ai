# Deployment — CropFresh AI (AWS)

> **Platform:** AWS App Runner (Phase 2–4) → ECS Fargate (Phase 5+)
> **Region:** ap-south-1 (Mumbai) — co-located with RDS, Redis, Bedrock

---

## Architecture

```
Client (Flutter / Buyer Dashboard)
  → AWS API Gateway (api.cropfresh.in / ws.cropfresh.in)
  → AWS App Runner (FastAPI, ECR image)
  → AWS RDS PostgreSQL (private subnet) + External services
```

Full diagram: see [`architecture.md §16`](../../_bmad-output/planning-artifacts/architecture.md)

---

## Environments

| Env        | API Base URL                       | Web URL                 | Deploy Trigger      |
| ---------- | ---------------------------------- | ----------------------- | ------------------- |
| Local      | `http://localhost:8000`            | `http://localhost:3000` | `docker compose up` |
| Staging    | `https://api-staging.cropfresh.in` | —                       | Push to `develop`   |
| Production | `https://api.cropfresh.in`         | `https://cropfresh.in`  | Push to `main`      |

---

## CI/CD Pipeline

Managed by `.github/workflows/deploy-aws.yml` with **GitHub OIDC** (no long-lived AWS keys stored in GitHub):

```
1. Push to GitHub (main)
   ↓
2. GitHub Actions — Job 1: Lint (ruff) + Tests (pytest)
   ↓
3. Job 2: docker build -f infra/docker/api.Dockerfile
          → push to AWS ECR (SHA tag + latest)
          → Trivy vulnerability scan
   ↓
4. Job 3: aws apprunner update-service
          → wait for RUNNING state
          → GET https://api.cropfresh.in/health (10 retries × 10s)
   ↓
5. Job 4: aws s3 sync web/out/ s3://cropfresh-web-prod/
          → CloudFront invalidation /* (buyer dashboard)
```

---

## Docker Build

```bash
# Build
docker build -f infra/docker/api.Dockerfile -t cropfresh-ai:latest .

# Run locally
docker run -p 8000:8000 --env-file .env cropfresh-ai:latest

# Push to ECR (done by GitHub Actions)
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin {ACCOUNT}.dkr.ecr.ap-south-1.amazonaws.com
docker push {ACCOUNT}.dkr.ecr.ap-south-1.amazonaws.com/cropfresh-ai:latest
```

---

## App Runner Configuration

File: `infra/aws/app-runner.yaml`

- **CPU:** 2 vCPU, **Memory:** 4 GB per instance
- **Auto-scale:** 1–10 instances, scale at 100 concurrent requests/instance
- **Peak pre-warm:** MinSize=2 during 04:30–14:30 IST (EventBridge Scheduler)
- **VPC Connector:** routes RDS traffic through private subnet

---

## Environment Variables (Production)

All secrets managed via **AWS Secrets Manager** (`cropfresh/prod/*`).
Injected into App Runner instances at startup via `RuntimeEnvironmentSecrets`.

See `infra/aws/secrets-template.yaml` for the full list of secret names.

**GitHub Actions secrets** (stored in repo settings, not Secrets Manager):

- `AWS_ROLE_ARN` — `cropfresh-github-actions-role` ARN
- `AWS_REGION` — `ap-south-1`
- `APP_RUNNER_SERVICE_ARN` — App Runner service ARN
- `CLOUDFRONT_DISTRIBUTION` — CloudFront distribution ID

---

## Monitoring

- **App Runner built-in metrics** — CPU, memory, request count, latency (CloudWatch)
- **Custom dashboards** — Grafana (Phase 5, `infra/monitoring/grafana/`)
- **Structured logs** — Loguru → CloudWatch Logs (`/aws/apprunner/cropfresh-ai`)
- **LLM tracing** — LangSmith (active)

---

## Useful AWS CLI Commands

```bash
# View App Runner service status
aws apprunner describe-service \
  --service-arn {APP_RUNNER_SERVICE_ARN} \
  --query 'Service.Status'

# Manually trigger a deployment
aws apprunner start-deployment \
  --service-arn {APP_RUNNER_SERVICE_ARN}

# View recent CloudWatch logs
aws logs tail /aws/apprunner/cropfresh-ai --follow

# List secrets
aws secretsmanager list-secrets \
  --filter Key=name,Values=cropfresh/prod

# Check RDS instance status
aws rds describe-db-instances \
  --db-instance-identifier cropfresh-db \
  --query 'DBInstances[0].DBInstanceStatus'
```

---

## Local Development

```bash
# 1. Copy env file
cp .env.example .env
# Fill in: GROQ_API_KEY, QDRANT_API_KEY, NEO4J_URI etc.

# 2. Start all services
docker compose up -d

# 3. Start API (hot reload)
uv run uvicorn src.api.main:app --reload --port 8000
```

Local services: FastAPI (8000), Redis (6379), Qdrant (6333), Neo4j (7474), Prometheus (9090), Grafana (3000)

---

## Deprecated

> GCP Cloud Run configuration and GCP Container Registry references have been archived to `infra/gcp/_deprecated/`.
> CropFresh AI is 100% AWS as of 2026-03-05.
