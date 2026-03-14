"""Generate App Runner config JSON with properly escaped secrets."""
import json
import subprocess
import os

def get_secret(name):
    r = subprocess.run(
        ['aws', 'secretsmanager', 'get-secret-value',
         '--secret-id', name, '--region', 'ap-south-1',
         '--query', 'SecretString', '--output', 'text', '--no-cli-pager'],
        capture_output=True, text=True
    )
    return r.stdout.strip()

config = {
    "ServiceName": "cropfresh-ai",
    "SourceConfiguration": {
        "ImageRepository": {
            "ImageIdentifier": "669640509011.dkr.ecr.ap-south-1.amazonaws.com/cropfresh-ai:latest",
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8000",
                "RuntimeEnvironmentVariables": {
                    "ENVIRONMENT": "production",
                    "LLM_PROVIDER": "groq",
                    "GROQ_API_KEY": get_secret("cropfresh/prod/groq_api_key"),
                    "DATABASE_URL": get_secret("cropfresh/prod/database_url"),
                    "REDIS_URL": get_secret("cropfresh/prod/redis_url"),
                    "QDRANT_API_KEY": get_secret("cropfresh/prod/qdrant_api_key"),
                    "JWT_SECRET": get_secret("cropfresh/prod/jwt_secret"),
                    "QDRANT_URL": "https://283826ed-234c-478e-820d-912d5a408a0f.eu-central-1-0.aws.cloud.qdrant.io:6333",
                    "QDRANT_COLLECTION": "agri_knowledge",
                }
            }
        },
        "AutoDeploymentsEnabled": True,
        "AuthenticationConfiguration": {
            "AccessRoleArn": "arn:aws:iam::669640509011:role/cropfresh-apprunner-ecr-access"
        }
    },
    "InstanceConfiguration": {
        "Cpu": "1 vCPU",
        "Memory": "2 GB",
        "InstanceRoleArn": "arn:aws:iam::669640509011:role/cropfresh-app-runner-role"
    },
    "HealthCheckConfiguration": {
        "Protocol": "HTTP",
        "Path": "/health",
        "Interval": 20,
        "Timeout": 10,
        "HealthyThreshold": 1,
        "UnhealthyThreshold": 10
    }
}

outpath = os.path.join(os.environ.get("TEMP", os.environ.get("TMP", ".")), "apprunner.json")
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False)

db = config["SourceConfiguration"]["ImageRepository"]["ImageConfiguration"]["RuntimeEnvironmentVariables"]["DATABASE_URL"]
print(f"Config written to {outpath}")
print(f"DB URL length: {len(db)}")

# Now create the service
import sys
result = subprocess.run(
    ['aws', 'apprunner', 'create-service',
     '--cli-input-json', f'file://{outpath}',
     '--region', 'ap-south-1',
     '--no-cli-pager',
     '--query', 'Service.{Status:Status,Url:ServiceUrl}',
     '--output', 'json'],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(f"ERROR: {result.stderr}")
    sys.exit(1)
