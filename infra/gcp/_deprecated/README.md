# ⚠️ DEPRECATED — GCP Infrastructure Files

**Status:** Archived on 2026-03-05  
**Reason:** CropFresh AI migrated to 100% AWS deployment.

The files in this directory (`cloud-run.yaml`, `cloud-scheduler.yaml`, `secrets.yaml.example`) previously targeted GCP Cloud Run but were replaced by the AWS deployment stack because:

1. All data services (RDS PostgreSQL, Redis Cache, Bedrock LLM) are in AWS `ap-south-1`
2. The `/voice` WebSocket endpoint requires persistent connections (Lambda/Cloud Run cold starts incompatible)
3. Single-cloud simplifies networking, reduces egress costs, and unifies IAM

## Replacement

| Old (GCP)                         | New (AWS)                                                           |
| --------------------------------- | ------------------------------------------------------------------- |
| `infra/gcp/cloud-run.yaml`        | `infra/aws/app-runner.yaml`                                         |
| GCP Secret Manager                | `infra/aws/secrets-template.yaml` → AWS Secrets Manager             |
| GCP Container Registry (`gcr.io`) | AWS ECR (`{ACCOUNT}.dkr.ecr.ap-south-1.amazonaws.com`)              |
| Cloud Scheduler                   | APScheduler (in-process) + EventBridge Scheduler (MinSize pre-warm) |
| GitHub Actions → GCP              | `.github/workflows/deploy-aws.yml`                                  |

See `docs/architecture/deployment.md` and `architecture.md §16` for the complete AWS deployment plan.
