# AWS VPC & Networking — CropFresh AI

Region: `ap-south-1` (Mumbai)

---

## VPC Layout

```
VPC: cropfresh-vpc
CIDR: 10.0.0.0/16

Availability Zones: ap-south-1a, ap-south-1b (2 AZs for RDS Multi-AZ at Phase 5)

┌────────────────────────────────────────────────────────────────┐
│  cropfresh-vpc  (10.0.0.0/16)                                  │
│                                                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │   Public Subnet 1a      │  │   Public Subnet 1b          │  │
│  │   10.0.1.0/24           │  │   10.0.2.0/24               │  │
│  │   (NAT Gateway)         │  │   (NAT Gateway standby)     │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │   Private Subnet 1a     │  │   Private Subnet 1b         │  │
│  │   10.0.11.0/24          │  │   10.0.12.0/24              │  │
│  │   RDS PostgreSQL        │  │   RDS standby (Phase 5)     │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
│                                                                 │
│  App Runner VPC Connector → attached to private subnets        │
│  (App Runner itself is serverless, runs outside VPC)           │
└────────────────────────────────────────────────────────────────┘

Internet Gateway → Public Subnets
NAT Gateway      → Private Subnets outbound (Groq/Bedrock API calls)
```

---

## Security Groups

### `cropfresh-rds-sg`

Attached to: AWS RDS PostgreSQL

| Direction | Protocol | Port | Source                   | Purpose          |
| --------- | -------- | ---- | ------------------------ | ---------------- |
| Inbound   | TCP      | 5432 | `cropfresh-apprunner-sg` | App Runner → RDS |
| Outbound  | All      | All  | —                        | —                |

```bash
aws ec2 create-security-group \
  --group-name cropfresh-rds-sg \
  --description "CropFresh RDS - allow App Runner only" \
  --vpc-id vpc-XXXXX
```

### `cropfresh-apprunner-sg`

Attached to: App Runner VPC Connector

| Direction | Protocol | Port  | Source             | Purpose                                       |
| --------- | -------- | ----- | ------------------ | --------------------------------------------- |
| Outbound  | TCP      | 5432  | `cropfresh-rds-sg` | App Runner → RDS                              |
| Outbound  | TCP      | 443   | 0.0.0.0/0          | Groq, Bedrock, Qdrant, Neo4j, Redis API calls |
| Outbound  | TCP      | 13641 | 0.0.0.0/0          | Redis Cloud (port 13641)                      |

---

## App Runner VPC Connector

```bash
aws apprunner create-vpc-connector \
  --vpc-connector-name cropfresh-vpc-connector \
  --subnets subnet-PRIVATE-1a subnet-PRIVATE-1b \
  --security-groups sg-APPRUNNER-ID
```

The VPC Connector attaches App Runner instances to private subnets, giving them:

- Access to RDS PostgreSQL (private subnet, no public exposure)
- Outbound internet via NAT Gateway for external API calls (Groq, Bedrock, Qdrant, Redis Cloud, Neo4j)

---

## RDS PostgreSQL Subnet Group

```bash
aws rds create-db-subnet-group \
  --db-subnet-group-name cropfresh-rds-subnet-group \
  --db-subnet-group-description "CropFresh RDS private subnets" \
  --subnet-ids subnet-PRIVATE-1a subnet-PRIVATE-1b
```

---

## Route 53 DNS Entries

| Record             | Type      | Value                                   | TTL |
| ------------------ | --------- | --------------------------------------- | --- |
| `api.cropfresh.in` | A (Alias) | API Gateway HTTP API custom domain      | —   |
| `ws.cropfresh.in`  | A (Alias) | API Gateway WebSocket API custom domain | —   |
| `cropfresh.in`     | A (Alias) | CloudFront distribution (web frontend)  | —   |
| `www.cropfresh.in` | CNAME     | `cropfresh.in`                          | 300 |

---

## Cost Estimate

| Component                   | Est. Monthly                |
| --------------------------- | --------------------------- |
| NAT Gateway (1x, Phase 2–4) | ~$35                        |
| App Runner VPC Connector    | $0 (included in App Runner) |
| Route 53 Hosted Zone        | ~$0.50                      |
| **Total Networking**        | **~$35/mo**                 |

> ⚠️ NAT Gateway is the most expensive networking component. At Phase 5, evaluate replacing with VPC Endpoints for AWS services (S3, Secrets Manager, Bedrock) to reduce NAT traffic costs.
