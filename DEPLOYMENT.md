# 🚀 CropFresh AI — AWS Deployment Tracker

> **Last Updated:** 2026-03-07 12:57 IST
> **Region:** `ap-south-1` (Mumbai)
> **AWS Account:** `669640509011`

---

## Architecture

```
┌──────────────────────────────────┐                    ┌───────────────────────────────────┐
│  AWS App Runner (CPU)            │   Internal API     │  EC2 g4dn.xlarge (GPU)            │
│  ──────────────────────────      │  ──────────────►   │  ─────────────────────────────    │
│  FastAPI + 15 AI Agents          │  POST /infer/*     │  ML Inference Server (:8001)      │
│  EdgeTTS (cloud TTS fallback)    │                    │  • IndicTTS (Parler TTS)           │
│  sentence-transformers (embeds)  │                    │  • faster-whisper STT              │
│  ~2GB Docker image (CPU torch)   │                    │  • Vision ONNX (YOLOv2, DINOv2)   │
│  ~$25/mo                         │                    │  • Silero VAD                      │
│                                  │                    │  ~$420/mo (24/7)                   │
└──────────┬───────────────────────┘                    └──────────┬────────────────────────┘
           │                                                       │
           ├──── RDS PostgreSQL ───────────────────────────────────┤
           ├──── Redis Cloud ──────────────────────────────────────┤
           ├──── Qdrant Cloud ─────────────────────────────────────┤
           └──── Secrets Manager ──────────────────────────────────┘
```

---

## Phase 1: Docker Build & ECR Push ✅

| Step                 | Status | Details                                                             |
| -------------------- | ------ | ------------------------------------------------------------------- |
| Root cause analysis  | ✅     | `sentence-transformers` → `torch` → CUDA (5GB+)                     |
| CPU-only Dockerfile  | ✅     | System pip installs CPU torch, `uv sync` skips 16 CUDA pkgs         |
| GitHub Actions build | ✅     | Built in **2m52s** on Ubuntu runners                                |
| ECR push             | ✅     | `669640509011.dkr.ecr.ap-south-1.amazonaws.com/cropfresh-ai:latest` |

**Key files:**

- [`infra/docker/api.Dockerfile`](file:///d:/Cropfresh-dev/Cropfresh%20Ai/Cropfresh_Ai/infra/docker/api.Dockerfile)
- [`.github/workflows/deploy-aws.yml`](file:///d:/Cropfresh-dev/Cropfresh%20Ai/Cropfresh_Ai/.github/workflows/deploy-aws.yml)

---

## Phase 2: App Runner Deployment ⏳

| Step                   | Status             | Details                                      |
| ---------------------- | ------------------ | -------------------------------------------- |
| GitHub CLI installed   | ✅                 | v2.87.3, authenticated as `cropfresh`        |
| AWS secrets in GitHub  | ✅                 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| 1st App Runner create  | ❌ Failed          | Missing env vars → container crash           |
| Failed service deleted | ✅                 | Cleaned up                                   |
| 2nd App Runner create  | ⏳ **In Progress** | All env vars configured                      |

**Service URL:** `https://xjivm4x5cn.ap-south-1.awsapprunner.com`

**Environment Variables Set:**
| Var | Source |
|---|---|
| `ENVIRONMENT` | `production` |
| `LLM_PROVIDER` | `groq` |
| `GROQ_API_KEY` | Secrets Manager |
| `DATABASE_URL` | Secrets Manager → RDS |
| `REDIS_URL` | Secrets Manager → Redis Cloud |
| `QDRANT_API_KEY` | Secrets Manager |
| `QDRANT_URL` | Qdrant Cloud (`eu-central-1`) |
| `JWT_SECRET` | Secrets Manager |

**Check status:**

```bash
aws apprunner list-services --region ap-south-1 --query "ServiceSummaryList[*].{Name:ServiceName,Status:Status}"
```

**Test health:**

```bash
curl https://xjivm4x5cn.ap-south-1.awsapprunner.com/health
```

---

## Phase 3: EC2 GPU Instance ⏳

| Step              | Status             | Details                                                |
| ----------------- | ------------------ | ------------------------------------------------------ |
| AMI selected      | ✅                 | `ami-0486051e45ffb9080` — DL PyTorch 2.8, Ubuntu 24.04 |
| Key pair created  | ✅                 | `cropfresh-gpu-key` (saved to `~/.ssh/`)               |
| Security group    | ✅                 | `sg-0318fe57d83008dab` — SSH(22) + inference(8001)     |
| GPU quota request | ⏳ **CASE_OPENED** | 0 → 4 vCPUs for G/VT instances                         |
| Launch instance   | ⬜ Blocked         | Waiting for quota approval                             |
| Setup ML server   | ⬜                 | FastAPI inference server on :8001                      |
| Download models   | ⬜                 | IndicTTS, faster-whisper, vision ONNX                  |

**Check quota status:**

```bash
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region ap-south-1 \
  --query "RequestedQuotas[*].{Quota:QuotaName,Status:Status,Desired:DesiredValue}"
```

---

## Phase 4: End-to-End Verification ⬜

- [ ] `GET /health` → `{"status": "alive"}`
- [ ] `GET /health/ready` → all checks pass
- [ ] `POST /api/v1/chat` → agent responds
- [ ] `POST /api/v1/voice/synthesize` → EdgeTTS audio
- [ ] EC2 GPU: IndicTTS generates speech
- [ ] EC2 GPU: faster-whisper transcribes audio
- [ ] EC2 GPU: Vision ONNX detects defects

---

## AWS Resources Created

| Resource            | ID / ARN                 | Monthly Cost |
| ------------------- | ------------------------ | ------------ |
| ECR Repository      | `cropfresh-ai`           | ~$1          |
| RDS PostgreSQL      | `cropfresh-ai-db-mumbai` | ~$15         |
| App Runner          | `cropfresh-ai`           | ~$25         |
| EC2 g4dn.xlarge     | _(pending quota)_        | ~$420 (24/7) |
| Secrets Manager     | 8 secrets                | ~$3          |
| **Total Estimated** |                          | **~$464/mo** |

---

## Quick Commands

```bash
# Check App Runner
aws apprunner list-services --region ap-south-1 --query "ServiceSummaryList[*].{Name:ServiceName,Status:Status}"

# Check GPU quota
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region ap-south-1 \
  --query "RequestedQuotas[*].{Quota:QuotaName,Status:Status}"

# Re-trigger GitHub Actions deployment
gh workflow run "Deploy CropFresh AI to AWS" --repo cropfresh/Cropfresh_Ai --ref main

# SSH to GPU instance (after launch)
ssh -i ~/.ssh/cropfresh-gpu-key.pem ubuntu@<GPU_PUBLIC_IP>

# ECR login
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 669640509011.dkr.ecr.ap-south-1.amazonaws.com
```
