# AWS S3 Buckets — CropFresh AI

Region: `ap-south-1` (Mumbai) — co-located with RDS and Redis for minimal latency.

---

## Bucket 1: `cropfresh-photos-prod`

**Purpose:** Farmer crop photos uploaded via `/vision/analyze` for quality assessment.

```yaml
BucketName: cropfresh-photos-prod
Region: ap-south-1
Versioning: Disabled # photos are unique per upload

LifecycleRules:
  - ID: archive-old-photos
    Status: Enabled
    Transitions:
      - Days: 30
        StorageClass: STANDARD_IA # infrequent access after 30 days
      - Days: 365
        StorageClass: GLACIER # archive after 1 year
    Expiration:
      Days: 1825 # delete after 5 years

CorsConfiguration:
  CORSRules:
    - AllowedOrigins: ["https://cropfresh.in"]
      AllowedMethods: [GET, PUT, POST]
      AllowedHeaders: ["*"]
      MaxAgeSeconds: 3000

PublicAccess: BLOCKED # private; accessed via pre-signed URLs


# Folder structure:
# cropfresh-photos-prod/
#   {farmer_id}/
#     {listing_id}/
#       departure_{timestamp}.jpg
#       arrival_{timestamp}.jpg
```

**Access Pattern:** App Runner generates pre-signed URLs (15 min TTL) for secure farmer uploads directly from the Flutter app. The URL is returned by `/vision/analyze` before processing.

---

## Bucket 2: `cropfresh-models-prod`

**Purpose:** ONNX model weights for Vision AI pipeline (YOLOv26, DINOv2, ResNet50).

```yaml
BucketName: cropfresh-models-prod
Region: ap-south-1
Versioning: Enabled # track model versions

LifecycleRules:
  - ID: expire-old-versions
    Status: Enabled
    NoncurrentVersionExpiration:
      NoncurrentDays: 90 # keep last 90 days of old versions

PublicAccess: BLOCKED # private only


# Folder structure:
# cropfresh-models-prod/
#   yolo/
#     yolov26_crop_defects_v{N}.onnx
#   dinov2/
#     dinov2_vit_s14_grade_v{N}.onnx
#   resnet50/
#     resnet50_digital_twin_v{N}.onnx
#   manifests/
#     current.json    ← { "yolo": "v3", "dinov2": "v2", "resnet50": "v1" }
```

**Load Strategy:** At App Runner startup, `vision_models.py` reads `manifests/current.json` and downloads required ONNX files to `/tmp/models/`. Cached for the instance lifetime. Cold start adds ~15–30s (first download). Subsequent requests use cached models.

---

## Bucket 3: `cropfresh-web-prod`

**Purpose:** Buyer dashboard static web app (React/Next.js exported static files).

```yaml
BucketName: cropfresh-web-prod
Region: ap-south-1
Versioning: Enabled

WebsiteConfiguration:
  IndexDocument: index.html
  ErrorDocument: index.html # SPA: all 404s → index.html for React Router

PublicAccess: BLOCKED # served only via CloudFront (OAC policy)
BucketPolicy: CloudFrontOAC # only CloudFront Origin Access Control can read


# Folder structure:
# cropfresh-web-prod/
#   _next/static/      ← Next.js static assets (long cache TTL)
#   images/            ← Static images
#   index.html         ← Entry point
```

**CloudFront Distribution:** Configured with `cropfresh-web-prod` as origin. Cache behaviour:

- `/_next/static/*` → TTL 31536000 (1 year, content-hashed filenames)
- `/*.html` → TTL 0 (no cache, always fresh)
- Default → TTL 86400 (1 day)

---

## IAM Access Summary

| Principal                       | Bucket                  | Permissions                                                          |
| ------------------------------- | ----------------------- | -------------------------------------------------------------------- |
| `cropfresh-app-runner-role`     | `cropfresh-photos-prod` | `s3:PutObject`, `s3:GetObject`, `s3:GeneratePresignedUrl`            |
| `cropfresh-app-runner-role`     | `cropfresh-models-prod` | `s3:GetObject` only                                                  |
| `cropfresh-github-actions-role` | `cropfresh-web-prod`    | `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket` (for deployments) |
| CloudFront OAC                  | `cropfresh-web-prod`    | `s3:GetObject` only                                                  |

---

## Cost Estimate

| Bucket                  | Storage            | Requests       | Est. Monthly |
| ----------------------- | ------------------ | -------------- | ------------ |
| `cropfresh-photos-prod` | ~50GB/month growth | ~10K PUT/month | ~$3–5        |
| `cropfresh-models-prod` | ~2GB static        | ~100 GET/month | ~$0.05       |
| `cropfresh-web-prod`    | ~50MB static       | Via CloudFront | ~$0.01       |
| **Total S3**            |                    |                | **~$3–6/mo** |
